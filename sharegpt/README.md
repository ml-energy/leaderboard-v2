# How we used ShareGPT to create our benchmark dataset

## Download ShareGPT
```
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
```

## Install Fastchat
```
pip install fschat
```

## Clean data:
```
pip install polyglot pyicu pycld2
python -m fastchat.data.optional_clean --in sg_90k_part1_html_cleaned.json --out sg_90k_part1_html_cleaned_lang.json --keep-lang en
```

## Extract first prompt
```
python extract_first.py --in-file sg_90k_part1_html_cleaned_lang.json --out-file sg_90k_part1_html_cleaned_lang_first.json
```

## Sample data
```
python -m fastchat.data.sample --in sg_90k_part1_html_cleaned_lang_first.json --out sg_90k_part1_html_cleaned_lang_first_sampled.json --end 10000 --max-length 10000
```

## Sorted data
'''
python sort.py --data-dir sg_90k_part1_html_cleaned_lang_first_sampled.json --out-file sg_90k_part1_html_cleaned_lang_first_sampled_sorted.json
'''
