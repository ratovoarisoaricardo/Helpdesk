import torch
from .data_utils import normalizeString, tensorFromSentence, SOS_token, EOS_token

def evaluate(encoder, decoder, vocab, sentence, max_length=15):
    with torch.no_grad():
        sentence = normalizeString(sentence)
        input_tensor = tensorFromSentence(vocab, sentence, max_length)
        input_tensor = input_tensor.view(1, -1)
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        decoder_input = torch.tensor([[SOS_token]], dtype=torch.long)
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        
        for di in range(max_length):
            decoder_output, decoder_hidden, _ = decoder.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            topv, topi = decoder_output.topk(1)
            
            # topi shape is (1, 1), extract item
            token = topi.item()
            
            if token == EOS_token:
                break
            else:
                # Ne pas inclure le padding dans la sortie
                word = vocab.index2word[token]
                if word not in ["<PAD>", "<SOS>", "<UNK>"]:
                    decoded_words.append(word)
                
            decoder_input = topi.squeeze(-1).detach() # prepare next input
            
        return ' '.join(decoded_words)
