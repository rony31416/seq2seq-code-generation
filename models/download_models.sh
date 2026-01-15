```bash
#!/bin/bash

echo "=================================================="
echo "Downloading Model Files from Google Drive"
echo "=================================================="

# Your Google Drive file IDs (get from sharing link)
VANILLA_RNN_ID="YOUR_FILE_ID_1"
LSTM_ID="YOUR_FILE_ID_2"
LSTM_ATTENTION_ID="YOUR_FILE_ID_3"

# Download function
download_file() {
    FILE_ID=$1
    OUTPUT=$2
    
    echo "Downloading $OUTPUT..."
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILE_ID" -O "$OUTPUT"
}

# Create models directory
mkdir -p models

# Download models
download_file $VANILLA_RNN_ID "models/vanilla_rnn_best.pt"
download_file $LSTM_ID "models/lstm_best.pt"
download_file $LSTM_ATTENTION_ID "models/lstm_attention_best.pt"

echo "=================================================="
echo " All models downloaded successfully!"
echo "=================================================="
```
