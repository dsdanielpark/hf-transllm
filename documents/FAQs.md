# Frequently Asked Questions (FAQs)

#### `ImportError: cannot import name 'LLaMATokenizer' from 'transformers'`
https://stackoverflow.com/questions/75907910/importerror-cannot-import-name-llamatokenizer-from-transformers

<br>

#### `Pytorch matmul - RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'`
If map==gpu, torch_dtype=torch.float16(default), if map==cpu, torch_dtype=torch.float32.

https://stackoverflow.com/questions/73530569/pytorch-matmul-runtimeerror-addmm-impl-cpu-not-implemented-for-half
```python
open_llama_kor = LLMtranslator(model_path, target_lang='ko', torch_dtype=torch.float32, translator='google')
```

<br>

#### `ValueError: The current device_map had weights offloaded to the disk. #239`
Using offload_folder args. (default)
https://github.com/nomic-ai/gpt4all/issues/239

<br>