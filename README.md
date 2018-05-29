# Sequence-to-Sequence-Models
Neural Sequence to Sequence Modeling
Currently Two seq2seq models are implemented. All performance was measured on Penn Tree Bank Dataset for POS Tagging.
# seq2seq_model_1.py
This model uses pre-trained word vectors (glove).

Model architecture is Embedding ---> BILSTM --->Softmax

Performance : 95.48% (Accuracy)
# seq2seq_with_char_embed.py
Main architecture is the same: Embedding-->BILSTM--->Softmax.

However, this model adds character embeddings.

Performance : 97.33% (Accuracy)
# Models will be added in future

# Dependencies
python 3.x

Pytorch 0.4
