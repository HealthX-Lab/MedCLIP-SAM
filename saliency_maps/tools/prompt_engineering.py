# %%
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    # 'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    # 'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    # 'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    # 'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    # 'art of the {}.',
    'a drawing of the {}.',
    # 'a photo of the large {}.',
    'a black and white photo of a {}.'
    'the plushie {}.',
    'a dark photo of a {}.',
    # 'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

SELECTED_TEMPLATES = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

from nltk.corpus import wordnet

def get_imagenet_prompts(reference, topk=7):
    return ([
            template.replace("{}", reference)
            for idx, template in enumerate(SELECTED_TEMPLATES)
            if idx <= topk
        ] if topk <= 7 else 
            [
            template.replace("{}", reference)
            for idx, template in enumerate(SELECTED_TEMPLATES)
            if idx <= topk
        ]
         +
        [
            template.replace("{}", reference)
            for idx, template in enumerate(IMAGENET_TEMPLATES)
            if idx < topk-7
        ])

def get_synonyms(word):
    return (
        [
            synonym
            for synonym in wordnet.synsets(word)[0].lemma_names()
            if synonym != word
        ]
        if wordnet.synsets(word)
        else []
    )

def get_engineered_prompts(reference, prompt_length, synonym_frist=True, keep_org=True, openai_templates=7):
    if synonym_frist:
         prompts = get_synonyms(reference) + get_imagenet_prompts(reference, prompt_length-len(get_synonyms(reference)))
    else:
        synonyms       = get_synonyms(reference)[:prompt_length-openai_templates - 1]
        openai_prompts = get_imagenet_prompts(reference, prompt_length-1-len(synonyms))
        prompts = openai_prompts + synonyms

    if keep_org:
        prompts.insert(0, reference) 
    return prompts