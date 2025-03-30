# GIT-LFS
To allow us to easily share and use our datasets. [git-lfs](https://git-lfs.com/) is used to circumvent git's file size limit      
Please set up git-lfs before commiting/pushing to the repo!

# news-bias-detection
## [Bias Detection](./BiasDetection/)
### [Sentiment](./BiasDetection/Sentiment/)
#### [pretrained_RoBERTa.py](./BiasDetection/Sentiment/pretrained_RoBERTa.py)
> [!IMPORTANT]
> Use Python version 3.11
> 
To check which Python version you are running use: 
```sh
which python3
python3 --version
```

Setting up the environment:
1. **Create venv with python 3.11** 

```sh
python3.11 -m venv venv
```

2. **Ensure you're inside the virtual environment.**
- Check if the venv folder has been created in your project directory.

3. **Activate the virtual environment**
For Mac: 
```sh 
source venv/bin/activate
```
For Windows:
```sh
venv\Scripts\activate
```
4. **Install the following packages:**
```sh
pip install nltk NewsSentiment==1.2.28
```
5. Install nltk resource:
```sh
python -c "import nltk; nltk.download('punkt')"
```

6. You're ready!