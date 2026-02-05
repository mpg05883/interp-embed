import re
import warnings
from functools import partial
import time
import numpy as np
import torch
from sae_lens import SAE as SAEModel
from scipy.sparse import csr_matrix
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.utils.path import resolve_model_snapshot

from .base_sae import BaseSAE, SAEType
from .utils import (
    ensure_loaded,
    get_goodfire_config,
    goodfire_sae_loader,
    store_activations_hook,
    try_to_load_feature_labels,
)

CONTEXT_WINDOW_LIMIT = 2048  # Context window limit used in the paper


class LocalSAE(BaseSAE):
    def __init__(
        self,
        sae_id="blocks.8.hook_resid_pre",
        release="gpt2-small-res-jb",
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.sae_id = sae_id
        self.release = release
        self.model = None
        self.sae = None
        self.tokenizer = None

    @property
    def name(self):
        cleaned_release = self.release.replace("/", "__")
        cleaned_sae_id = self.sae_id.replace("/", "__")
        return f"local__{cleaned_release}_{cleaned_sae_id}"

    def metadata(self):
        parent_metadata = super().metadata()
        parent_metadata.update(
            {
                "sae_id": self.sae_id,
                "release": self.release,
                "device": {"model": self.model_device, "sae": self.sae_device},
                "sae_type": SAEType.LOCAL,
            }
        )
        return parent_metadata

    def load_models(self):
        from transformer_lens import HookedTransformer

        print("Loading SAE...")
        self.sae = SAEModel.from_pretrained(
            release=self.release,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=self.sae_id,  # won't always be a hook point
            device=self.sae_device,
        )
        print("Loading language model...")
        self.model = HookedTransformer.from_pretrained(
            self.sae.cfg.metadata.model_name, device=self.model_device
        )
        self.tokenizer = self.model.tokenizer

    @ensure_loaded
    @torch.no_grad()
    def encode(self, texts):
        assert len(texts) > 0, "There must be more t.han one text to encode."
        self.sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
        
        print(f"self.model exists: {self.model is not None}")
        print(f"self.sae exists: {hasattr(self, 'sae') and self.sae is not None}")
        print(f"self.tokenizer exists: {self.tokenizer is not None}")

        # # Filter out texts that exceed the context window
        # max_length = self.tokenizer.model_max_length or CONTEXT_WINDOW_LIMIT
        # valid_texts = [text for text in texts if len(self.tokenizer.tokenize(text)) <= max_length]

        # if len(valid_texts) < len(texts):
        #     warnings.warn(f"{len(texts) - len(valid_texts)} texts were skipped because they exceed the context window of {max_length} tokens.")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokens = self.tokenizer(
            # valid_texts,
            texts,
            padding="longest",
            # truncation=True,
            truncation=self.truncate,
            # max_length=max_length,
            return_tensors="pt",
        )

        _, cache = self.model.run_with_cache(tokens["input_ids"], prepend_bos=True)

        # Use the SAE
        feature_acts = self.sae.encode(
            cache[self.sae.cfg.metadata.hook_name].to(self.sae_device)
        )

        feature_acts_np = feature_acts.detach().cpu().numpy()
        attn_mask = tokens["attention_mask"].numpy().astype(bool)
        return [
            csr_matrix(feature_acts_np[i][attn_mask[i]])
            for i in range(feature_acts_np.shape[0])
        ]

    @ensure_loaded
    def encode_chat(self, chat_conversations):
        assert (
            self.chat_template_exists()
        ), "Chat template does not exist for this model's tokenizer"
        texts = [
            self.tokenizer.apply_chat_template(chat_conversation, tokenize=False)
            for chat_conversation in chat_conversations
        ]
        return self.encode(texts)

    def destroy_models(self):
        self.sae = None
        self.model = None


class GoodfireSAE(BaseSAE):
    def __init__(
        self,
        variant_name: str = "Llama-3.3-70B-Instruct-SAE-l50",
        quantize=False,
        **kwargs,
    ):
        """
        Args:
            variant_name: The name of the Goodfire SAE to use. NOTE: Goodfire
                only supports SAEs trained on Llama models. See here for a list
                of supported models: https://huggingface.co/goodfire/goodfire-sae-models
            quantize: Whether to quantize the language model.
            **kwargs: Additional arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self.variant_name = variant_name
        self.quantize = quantize
        self.activations = dict()
        self.activation_hook_handle = None
        self.config = get_goodfire_config(variant_name)

    @property
    def name(self):
        cleaned_variant_name = self.variant_name.replace("/", "__")
        return f"goodfire__{cleaned_variant_name}"

    def metadata(self):
        parent_metadata = super().metadata()
        parent_metadata.update(
            {
                "variant_name": self.variant_name,
                "quantize": self.quantize,
                "device": {"model": self.model_device, "sae": self.sae_device},
                "sae_type": SAEType.GOODFIRE,
            }
        )
        return parent_metadata

    def load_feature_labels(self):
        self._feature_labels = try_to_load_feature_labels(
            self.config["feature_labels_file"]
        )

        # Load the feature labels
        if self._feature_labels:
            self._feature_labels = {
                int(key): value for key, value in self.feature_labels().items()
            }  # Convert keys to ints

    def load_models(self):
        # Load the model, sae, and tokenizer
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float32,
        )

        if self.quantize:
            warnings.warn(
                "Quantizing the language model may cause feature activations to be less accurate."
            )

        config = get_goodfire_config(self.variant_name)
        snapshot = resolve_model_snapshot(config["hf_model"])
        
        print(
            f"CUDA available: {torch.cuda.is_available()}, "
            f"GPU count: {torch.cuda.device_count()}"
        )
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,  
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if self.quantize else None,
            device_map=self.model_device,
            use_safetensors=True,
            # max_memory={0: "38GB", 1: "38GB", 2: "38GB", 3: "38GB"},
        )
        print(f"Model loaded in {time.time() - start:.1f}s")
        print(f"model map: {set(self.model.hf_device_map.values())}")

        # Add hooks to the model
        self.activations = {}
        match = re.search(r"l(\d+)", self.variant_name)
        if match is None:
            raise ValueError(
                f"Could not find layer number in filename: {self.variant_name}"
            )
        layer = int(match.group(1))
        activation_hook = partial(
            store_activations_hook, activations=self.activations, name="internal"
        )
        self.model.model.layers = torch.nn.ModuleList(
            self.model.model.layers[: layer + 1]
        )  # Truncate the model to the layer we want to extract activations from
        self.activation_hook_handle = self.model.model.layers[
            layer
        ].register_forward_hook(activation_hook)
        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            use_fast=True,
        )
        
        # NOTE: Use hard-coded release and sae_id from here:
        # https://decoderesearch.github.io/SAELens/latest/pretrained_saes/meta-llama_llama-3.3-70b-instruct/
        
        start = time.time()
        self.sae = SAEModel.from_pretrained(
            release="goodfire-llama-3.3-70b-instruct",
            sae_id="layer_50",
            device=self.sae_device,
            converter=goodfire_sae_loader,
        )
        print(f"SAE loaded in {time.time() - start:.1f}s")

        self.tokenizer.pad_token = self.tokenizer.eos_token

    @ensure_loaded
    def encode(self, texts):
        input_device = next(self.model.parameters()).device
        # print(f"Model device: {input_device}, SAE device: {self.sae_device}")
        # print("Starting encode()...")
            
        inputs = self.tokenize(texts, padding=True, as_tokens=False)

        with torch.no_grad():
            outputs = self.model(
                input_ids=torch.tensor(inputs["input_ids"]).to(input_device),
                attention_mask=torch.tensor(inputs["attention_mask"]).to(input_device),
            )

            feature_acts = self.sae.encode(
                self.activations["internal"].to(self.sae.device)
            )

        feature_acts_np = feature_acts.float().detach().cpu().numpy()
        attn_mask = np.array(inputs["attention_mask"]).astype(bool)

        # Clean up memory
        del outputs, inputs
        torch.cuda.empty_cache()

        return [
            csr_matrix(feature_acts_np[i][attn_mask[i]])
            for i in range(feature_acts_np.shape[0])
    ]

    def destroy_models(self):
        self.activations = dict()
        self.activation_hook_handle.remove()
        self.model = None
        self.sae = None
