# GIT-LFS
To allow us to easily share and use our datasets. [git-lfs](https://git-lfs.com/) is used to circumvent git's file size limit      
Please set up git-lfs before commiting/pushing to the repo!

# news-bias-detection
## [Bias Detection](./BiasDetection/)
### [Sentiment](./BiasDetection/Sentiment/)
#### [pretrained_RoBERTa.py](./BiasDetection/Sentiment/pretrained_RoBERTa.py)
> [!IMPORTANT]
> Use Python version 3.11

Setting up the environment:
1. Create venv with python 3.11
2. Ensusre you're inside the virtual environment
3. Install the following packages:
```sh
pip install nltk NewsSentiment==1.2.28
```
4. Install nltk resource:
```sh
python -c "import nltk; nltk.download('punkt')"
```
5. You're ready!