# Frequently Asked Questions (FAQs)

#### `ImportError: cannot import name 'LLaMATokenizer' from 'transformers'`
https://stackoverflow.com/questions/75907910/importerror-cannot-import-name-llamatokenizer-from-transformers

<br>

#### `Pytorch matmul - RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'`
- map==gpu, torch_dtype=torch.float16(default)
- map==cpu, torch_dtype=torch.float32.

https://stackoverflow.com/questions/73530569/pytorch-matmul-runtimeerror-addmm-impl-cpu-not-implemented-for-half

<br>

**GPU inference with offloaded folder**
```python
open_llama_kor = LLMtranslator(model_path, target_lang='ko', torch_dtype=torch.float16, translator='google', offload_folder="offload")
```

**CPU inference with offloaded folder**
```python
open_llama_kor = LLMtranslator(model_path, target_lang='ko', torch_dtype=torch.float32, translator='google', offload_folder="offload")
```

<br>

#### `ValueError: The current device_map had weights offloaded to the disk. #239`
Using offload_folder args.
https://github.com/nomic-ai/gpt4all/issues/239

<br>
