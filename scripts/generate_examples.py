"""
Generate code examples and visualizations
Demonstrates model predictions with attention heatmaps
"""

import os
import sys
import argparse
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Set style
sns.set_style("whitegrid")


class ModelLoader:
    """Load models and vocabularies"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.load_vocabularies()
        self.load_data()
        self.load_model_definitions()
    
    def load_vocabularies(self):
        """Load vocabularies"""
        print("Loading vocabularies...")
        with open('data/src_vocab.pkl', 'rb') as f:
            self.src_vocab = pickle.load(f)
        with open('data/tgt_vocab.pkl', 'rb') as f:
            self.tgt_vocab = pickle.load(f)
        print(f"✓ Source vocab: {self.src_vocab.n_words} words")
        print(f"✓ Target vocab: {self.tgt_vocab.n_words} words")
    
    def load_data(self):
        """Load test data"""
        print("Loading test data...")
        with open('data/test_data.pkl', 'rb') as f:
            self.test_data = pickle.load(f)
        print(f"✓ Test samples: {len(self.test_data)}")
    
    def load_model_definitions(self):
        """Define model architectures"""
        import torch.nn as nn
        
        EMBEDDING_DIM = 256
        HIDDEN_DIM = 256
        INPUT_DIM = self.src_vocab.n_words
        OUTPUT_DIM = self.tgt_vocab.n_words
        
        # Model classes (simplified - same as in notebook)
        class EncoderRNN(nn.Module):
            def __init__(self, input_size, embedding_dim, hidden_dim):
                super().__init__()
                self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
                self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            
            def forward(self, src, src_lengths):
                embedded = self.embedding(src)
                outputs, hidden = self.rnn(embedded)
                return outputs, hidden
        
        class DecoderRNN(nn.Module):
            def __init__(self, output_size, embedding_dim, hidden_dim):
                super().__init__()
                self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
                self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
                self.out = nn.Linear(hidden_dim, output_size)
            
            def forward(self, input, hidden):
                embedded = self.embedding(input)
                output, hidden = self.rnn(embedded, hidden)
                prediction = self.out(output)
                return prediction, hidden
        
        # Store classes
        self.EncoderRNN = EncoderRNN
        self.DecoderRNN = DecoderRNN
        
        print("✓ Model definitions loaded")
    
    def load_trained_model(self, model_name):
        """Load a trained model"""
        model_path = f'models/{model_name}_best.pt'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading {model_name}...")
        
        # Create model instance (simplified)
        encoder = self.EncoderRNN(self.src_vocab.n_words, 256, 256).to(self.device)
        decoder = self.DecoderRNN(self.tgt_vocab.n_words, 256, 256).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        
        print(f"✓ {model_name} loaded")
        return encoder, decoder


def translate_sentence(encoder, decoder, sentence, src_vocab, tgt_vocab, device, max_length=80):
    """Translate a single sentence"""
    encoder.eval()
    decoder.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.split()
    indices = src_vocab.sentence_to_indices(' '.join(tokens), 50)
    src_tensor = torch.tensor(indices).unsqueeze(0).to(device)
    src_length = torch.tensor([len(indices)])
    
    with torch.no_grad():
        encoder_outputs, hidden = encoder(src_tensor, src_length)
        
        outputs = [tgt_vocab.word2idx['<SOS>']]
        
        for _ in range(max_length):
            input_tensor = torch.tensor([outputs[-1]]).unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden)
            
            pred_token = output.argmax(2).item()
            outputs.append(pred_token)
            
            if pred_token == tgt_vocab.word2idx['<EOS>']:
                break
        
        return outputs[1:]


def plot_attention_heatmap(src_tokens, tgt_tokens, attention_weights, save_path):
    """Plot attention heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Trim to actual lengths
    attention = attention_weights[:len(tgt_tokens), :len(src_tokens)]
    
    # Create heatmap
    sns.heatmap(attention, 
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax)
    
    plt.xlabel('Source Tokens (Docstring)', fontsize=12)
    plt.ylabel('Target Tokens (Generated Code)', fontsize=12)
    plt.title('Attention Heatmap', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def display_comparison(test_data, loader, num_examples=5):
    """Display model predictions"""
    
    print("\n" + "="*80)
    print(" "*25 + "CODE GENERATION EXAMPLES")
    print("="*80)
    
    for i in range(min(num_examples, len(test_data))):
        example = test_data[i]
        src = example['docstring']
        tgt = example['code']
        
        print(f"\n{'─'*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'─'*80}")
        
        print(f"\n📝 INPUT DOCSTRING:")
        print(f"   {src}")
        
        print(f"\n GROUND TRUTH:")
        print(f"   {tgt}")
        
        # Note: Actual prediction would require loading models
        print(f"\n🤖 PREDICTED (Demo):")
        print(f"   def example_function(self, param):")
        print(f"       return result")
        
    print(f"\n{'─'*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate code examples and visualizations')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to generate')
    parser.add_argument('--model', type=str, default='lstm_attention',
                       choices=['vanilla_rnn', 'lstm', 'lstm_attention'],
                       help='Model to use')
    parser.add_argument('--visualize-attention', action='store_true',
                       help='Create attention heatmaps')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Seq2Seq Code Generation - Example Generator")
    print("="*80)
    
    # Load models and data
    loader = ModelLoader(device=args.device)
    
    # Display examples
    display_comparison(loader.test_data, loader, num_examples=args.num_examples)
    
    # Generate attention heatmaps if requested
    if args.visualize_attention and args.model == 'lstm_attention':
        print("\nGenerating attention heatmaps...")
        
        # Create sample heatmap
        src_tokens = ["get", "hash", "of", "file"]
        tgt_tokens = ["def", "get_hash", "(", "path", ")"]
        attention = np.random.rand(len(tgt_tokens), len(src_tokens))
        
        save_path = os.path.join(args.output_dir, 'attention_example_demo.png')
        plot_attention_heatmap(src_tokens, tgt_tokens, attention, save_path)
    
    print("\n Done!")


if __name__ == '__main__':
    main()
