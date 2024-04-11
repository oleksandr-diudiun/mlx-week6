from collections import defaultdict
import io
from pathlib import Path
from typing import Dict, List
from sentencepiece import SentencePieceProcessor

# Lets tokenize using sentence piece
from sentencepiece import SentencePieceTrainer
import pandas as pd
import torch
from tqdm import tqdm

SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"


def get_image_id_captions_mapping(dataframe: pd.DataFrame) -> Dict[int, List[str]]:
    all_imageid_caption_mapping = defaultdict(list)

    for _, row in dataframe.iterrows():
        image_id = int(row["img_id"])
        captions = row["caption"]
        all_imageid_caption_mapping[image_id].extend(captions)
    return all_imageid_caption_mapping


def preprocess_captions_to_padded_tokens(
    sp_processor: SentencePieceProcessor,
    all_imageid_caption_mapping: Dict[int, List[str]],
    max_len: int = 50,
):
    ## Pad all the tokens to size 36 and ignore the image ids which have more than 36 tokens
    # Create a mapping for this.
    all_image_id_token_mapping = defaultdict(list)
    # Use all_imageid_caption_mapping
    n_skipped = 0
    skipped = []
    for image_id, captions in tqdm(all_imageid_caption_mapping.items()):
        for caption in captions:
            encoded = sp_processor.encode_as_ids(caption.lower())
            if (
                len(encoded) > max_len - 3
            ):  # -3 because we have SOS, EOS and an `_` char that sentence processor adds to the beginning to account for.
                n_skipped += 1
                skipped.append((image_id, caption))
                continue
            paddings = [PAD] * (
                max_len - (len(encoded) + 3)
            )  # -3 not -2 because we have an addtional `_` char that sentence processor adds to the beginning of
            sentence = (
                SOS + " " + caption.lower() + EOS + "".join(paddings)
            )  # Note the lower case. Make sure to do that during inference too.
            # print(len(encoded), len(paddings))
            newly_encoded = sp_processor.encode_as_ids(sentence)
            assert (
                len(newly_encoded) == max_len
            ), f"Length of encoded is {len(newly_encoded)}"  # everything should be 50 now.
            all_image_id_token_mapping[image_id].append(
                torch.tensor(newly_encoded, dtype=torch.long)
            )
    return all_image_id_token_mapping


def get_model(train, config, all_sentences, root_dir: Path, models_dir: Path):
    model = io.BytesIO()
    model_filename = root_dir / models_dir / "spm_10000_vocab_model_image_caption.model"
    if train:
        SentencePieceTrainer.train(
            sentence_iterator=(story.lower() for story in all_sentences),
            model_writer=model,
            vocab_size=config["V"],
            user_defined_symbols=[SOS, EOS, PAD],
            # max_sentence_length=4196,
            # model_type="BPE",
        )
        sp_processor = SentencePieceProcessor(model_proto=model.getvalue())
    else:
        sp_processor = SentencePieceProcessor(model_file=str(model_filename))
    return sp_processor, model


def save_model(model, models_dir: Path, root: Path):
    model_filename = root / models_dir / "spm_10000_vocab_model_image_caption.model"
    with open(model_filename, "wb") as f:
        print(".")
        f.write(model.getvalue())
