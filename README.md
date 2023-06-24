Development Status :: 3 - Alpha

# Python Package: hf-transllm

<p align="left">
<a><img alt="PyPI package" src="https://img.shields.io/badge/pypi-transllm-black"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fdsdanielpark%2Fhf-transllm&count_bg=%23000000&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
<a href="https://pypi.org/project/transllm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/transllm"></a>
</p>


> LLMtranslator translates and generates text in multiple languages.

![](assets/transllm.png)

### Introducing hf-transllm: Unlock the Power of Multilingual Exploration

Discover the hf-transllm package, a seamless integration of Hugging Face's inference module and translation APIs. Overcome limitations in retraining and evaluating large language models in different languages. Explore diverse results effortlessly, leveraging translation API services. Emphasizing versatility over efficiency, hf-transllm enables you to delve into the outcomes of Hugging Face's models in various languages.


<br>


<br>

## Installation
```
pip install transllm
```

If you wish to use the various features and CLI:
```
pip install git+https://github.com/dsdanielpark/hf-transllm.git
cd hf-transllm
pip install -r requirements.txt
```
There can be issues with various dependencies such as Hugging Face's Transformers, SentencePiece, Torch, and CUDA. Please set up the appropriate environment by searching online.
```bash
python main.py --hfmodel <openlm-research/open_llama_3b> --lang <ko> --translator <google>
```
<br>

<br>

## Usage 
Applying LLMs to the majority of Hugging Face repositories is generally feasible. However, it can be challenging to apply them to objects that require unique tokenizers or inference processes. In such cases, it is recommended to customize the usage by incorporating a translation module for prompts.

In other words, if you are familiar with the inference process or code from Hugging Face repositories, you can customize the translation object by adding a translation module before and after the known inference process or code.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1117ikGEmU4FncBDl1xCC2IhPPDOr75lX?usp=sharing) 

Google Trnaslator
- Support Languages: https://github.com/ssut/py-googletrans/blob/master/googletrans/constants.py
```python
from transllm import LLMtranslator

# Set huggingface repository
model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

# Get TransLLM Object
open_llama3b_kor = LLMtranslator(model_path, target_lang='ko', translator='google')

# Using Prompt in multi-language
prompt = "나와 내 동년배들이 좋아하는 뉴진스에 대해서 알려줘"
trnaslated_answer = open_llama3b_kor.generate(prompt)
print(trnaslated_answer)
```

<br>

DeepL
- Support Languages: https://www.deepl.com/pro/select-country?cta=header-pro-button/#developer

Open AI, Bard use pre-prompt for translation.
```python
from transllm import LLMtranslator

# Set huggingface repository
model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

# Choose Translate Service API: DeepL, OpenAI, Bard
open_llama3b_kor = LLMtranslator(model_path, target_lang='EN', translator='deepl', deepl_api='xxxxxxx') 
# open_llama3b_kor = LLMtranslator(model_path, target_lang='korean', translator='openai', openai_api='xxxxxxx', openai_model='gpt-3.5-turbo')
# open_llama3b_kor = LLMtranslator(model_path, target_lang='korean', translator='bard', bard_api='xxxxxxx')

# Using Prompt in multi-language
prompt = "나와 내 동년배들이 좋아하는 뉴진스에 대해서 알려줘"
trnaslated_answer = open_llama3b_kor.generate(prompt)
print(trnaslated_answer)
```

<br>


## Contributors

I would like to express my sincere gratitude for the contributions made by all the contributors.

<a href="https://github.com/dsdanielpark/hf-transllm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dsdanielpark/hf-transllm" />
</a>


<br>

## License
[MIT](https://opensource.org/license/mit/) <br>
I hold no legal responsibility; 
```
The MIT License (MIT)

Copyright (c) 2023 Minwoo Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Bugs and Issues
Sincerely grateful for any reports on new features or bugs. Your valuable feedback on the code is highly appreciated.

## Contacts
- Core maintainer: [Daniel Park, South Korea](https://github.com/DSDanielPark) <br>
- E-mail: parkminwoo1991@gmail.com <br>

## Reference 
[1] https://huggingface.co/docs/api-inference/index <br>
  
<br>
            

  
*Copyright (c) 2023 MinWoo Park, South Korea*<br>
