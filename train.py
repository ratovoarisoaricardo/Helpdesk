import sys
import os

# Ensure the root of the project is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch import optim

from src.model import EncoderRNN, AttnDecoderRNN
from src.data_utils import prepareData, tensorsFromPair

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Forward pass
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
    
    loss = 0
    for i in range(target_tensor.size(1)):
        loss += criterion(decoder_outputs[:, i, :], target_tensor[:, i])
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensor.size(1)

def main():
    filepath = 'data/faq.json'
    vocab, pairs = prepareData(filepath)

    hidden_size = 128
    learning_rate = 0.01
    max_length = 15

    encoder = EncoderRNN(vocab.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, vocab.n_words)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print("Modèle et optimiseurs prêts.")

    epochs = 50

    print("Début de l'entraînement sur CPU...")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for pair in pairs:
            input_tensor, target_tensor = tensorsFromPair(pair, vocab, max_length)
            input_tensor = input_tensor.view(1, -1)
            target_tensor = target_tensor.view(1, -1)
            
            loss = train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            total_loss += loss
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(pairs):.4f}")
            
    print("Entraînement terminé.")

    # Sauvegarde des poids du modèle
    os.makedirs('models', exist_ok=True)
    torch.save(encoder.state_dict(), 'models/encoder.pth')
    torch.save(decoder.state_dict(), 'models/decoder.pth')

    print("Poids sauvegardés dans le dossier 'models/'.")

if __name__ == "__main__":
    main()
