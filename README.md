# Local Host Project Page

## How To Use

### Requirements

Streamlit is required and can be installed via `pip install streamlit`
Errors might occur because some pages install other dependencies. If this happens, install the missing packages. These might include: `tensorflow, keras, and keras-nlp`.

### Usage

One all the requirements are installed, download the git repository with `git clone https://github.com/dashblack576/Project_Page.git` move all the files into the directory of your choice. Once the files are collected in one repository, simply run the run.bat file and the landing page should load on a local host. You may experience a problem at first from antivirus software, but streamlit is safe and this should pass. If the run file doesn't work, navigate to the given directory where the Project Page files are stored and run the command `streamlit run landing_page.py`. 

## Supported Projects

As for right now, only summarization is supported. I'm working on implementing my other projects soon, though.

# Summarization

To use the summarization model, a few steps will be necessary. First navigate to the Seq2SeqV1 git project page -- https://github.com/dashblack576/seq2seqV1. After cloning that repository and following all the download steps, move the Model directory and vocab.txt file into the summarization directory. After doing that, you should be able to access the summarization page.
