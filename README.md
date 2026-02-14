FinGAT — Financial Graph Attention Network

FinGAT is a PyTorch-based implementation of a Graph Attention Network (GAT) for financial stock prediction and top-K profitable stock recommendation.

The model captures:

 Temporal patterns from stock price data

 Relationships between stocks and sectors using graph attention

 Return prediction for ranking profitable stocks

 Project Structure
'FinGAT/
├── Feature_Extraction.ipynb   # Stock feature engineering
├── create_edge.ipynb          # Graph construction
├── train_new.py               # Model training script
├── parse_arg.py               # Argument handling
├── inner.npy / outer.npy      # Graph data
├── NIFTY50_category.csv       # Sector labels'

 Tech Stack

Python • PyTorch • NumPy • Pandas

This project demonstrates how Graph Neural Networks can model inter-stock relationships to improve financial prediction performance.
