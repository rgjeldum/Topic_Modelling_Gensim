import os
import math
import sys
import time
import json
import re

from pprint import pprint
import numpy as np
import warnings
import pyLDAvis

warnings.filterwarnings('ignore')
import gensim
import pyLDAvis.gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities


def createGensimObjects(nameOfDocument):

    # e.g., Data/Novels/Alice_in_Wonderland/Alice_in_Wonderland.txt
    print("\ncalling createGensimObjects() method ...")
    print("nameOfDocument=" + nameOfDocument)

    stopWordsFile = 'StopWords/stopwords.txt'
    localWebServerDir = '/Applications/MAMP/htdocs/gensim/'

    outputDictPath = nameOfDocument.replace("Data", "Output", 1)
    #print("\noutputDictPath=" + outputDictPath + "\n")

    # e.g., Output/Novels/Alice_in_Wonderland
    outputDictDir = os.path.dirname(outputDictPath)
    print("\noutputDictDir=" + outputDictDir + "\n")

    # remove common words and tokenize
    stoplist = set(line.strip().lower() for line in open(stopWordsFile))
    print("\nstopwords:")
    print(stoplist)

    fullPathName = nameOfDocument
    # e.g., Data/Novels/Alice_in_Wonderland/Alice_in_Wonderland.txt
    print("\nfullPathName=" + fullPathName)

    base_document_name = os.path.splitext(nameOfDocument)[0]
    # base_document_name = os.path.dirname(nameOfDocument)
    # e.g., Data/Novels/Alice_in_Wonderland/Alice_in_Wonderland
    print("base_document_name=" + base_document_name)

    outputFileDir = localWebServerDir + base_document_name
    print("\noutputFileDir=" + outputFileDir + "\n")

    base_dict_name = base_document_name.replace("Data", "Output", 1)
    print("\nbase_dict_name=" + base_dict_name)

    # newOutputFileDir = re.sub(outputFileDir, "\.\/Data", "\.\/Output")
    gensimOutputDir = outputFileDir.replace("Output/", "", 1)
    print("gensimOutputDir=" + gensimOutputDir + "\n")

    # gensimOutputDirStripped = os.path.dirname(os.path.dirname(gensimOutputDir))
    # print("gensimOutputDirStripped=" + gensimOutputDirStripped)

    # sys.exit(-20)

    class MyCorpus(object):
        def __iter__(self):
            for line in open(fullPathName):
                # assume there's one document per line, tokens separated by whitespace
                yield dictionary.doc2bow(line.lower().split())

    corpus = MyCorpus()  # doesn't load the corpus into memory!

    from six import iteritems
    from gensim import corpora

    # collect statistics about all tokens
    dictionary = corpora.Dictionary(line.lower().split() for line in open(fullPathName))

    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    ## RJG dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.filter_tokens(stop_ids)  # remove stop words but KEEP words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed

    dictName = base_dict_name + '.dict'
    print("\nsaving dictionary: " + dictName + "\n")
    dictionary.save(dictName)


    print("\ndictionary keys:\n")
    list(dictionary.keys())
    # for k,v in dictionary.items():
    #     print(k,v)


    mmName = base_dict_name + '.mm'
    print("\nsaving mm corpus: " + mmName)
    corpora.MmCorpus.serialize(mmName, corpus)

    ldacName = base_dict_name + '.lda-c'
    print("\nsaving lda-c corpus:" + ldacName)
    corpora.BleiCorpus.serialize(ldacName, corpus)

    # print(corpus)
    from gensim import corpora, models, similarities
    if (os.path.exists(dictName)):
        dictionary = corpora.Dictionary.load(dictName)
        corpus = corpora.BleiCorpus(ldacName)
    else:
        print("\nERROR - need to first generate gensim objects.\n")

    modelName = base_dict_name + '.model'
    print("\nsaving model: " + modelName + "\n")
    model = models.LdaModel(corpus, id2word=dictionary, num_topics=80)
    model.save(modelName)

    print("\nmodel:")
    print(model)

    dictionary = gensim.corpora.Dictionary.load(dictName)
    corpus = gensim.corpora.MmCorpus(mmName)
    lda = gensim.models.ldamodel.LdaModel.load(modelName)


    print("\ndictionary:\n")
    print(dictionary)

    print("\ncorpus:\n")
    print(corpus)

    print("\nlda:\n")
    print(lda)

    print("\n2 topics / 4 words:\n")
    print(lda.print_topics(num_topics=2, num_words=4))

    for i in lda.print_topics():
        for j in i: print(j)

    # create visualization
    # pyLDAvis.enable_notebook()
    # pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    visualisation = pyLDAvis.gensim.prepare(lda, corpus, dictionary)

    if("Novels" in gensimOutputDir):
        gensimOutputDir = gensimOutputDir.replace("Data/", "", 1)
        print("\ngensimOutputDir(revised)=" + gensimOutputDir + "\n")

    visFileName = gensimOutputDir + '.vis.html'
    print("saving visualization file to: "+ visFileName + "\n")
    pyLDAvis.save_html(visualisation, visFileName)

    # now copy vis file to htdocs directory so web browser can display it
    from shutil import copyfile
    outputFile =  visFileName
    print("outputFile=" + outputFile + "\n")

    visName = base_dict_name + '.vis.html'
    print("\ncopying visualization file to: "+ visName + "\n")

    copyfile(visFileName, visName)


if __name__ == '__main__':

    # get no. of args from command line
    argLength = len(sys.argv)
    if (argLength != 2):
        print("\nUsage: {} <dir name of \.txt file(s)>".format(sys.argv[0]))
        sys.exit(1)

    # e.g., /Data/Novels/Alice_in_Wonderland or /Data/MDA/2017
    inputDirectoryName = sys.argv[1]

    # Data/Novels/Alice_in_Wonderland
    print("\ndirectory where input files are located: " + inputDirectoryName)

    outputDirectoryName = inputDirectoryName.replace("Data", "Output", 1)

    # Output/Novels/Alice_in_Wonderland
    print("\ndirectory where output files are located: " + outputDirectoryName)

    # start timer for the topic modelling process
    start_time = time.time()

    # from os import walk
    # fileName = walk("Data").next()
    # print(fileName)

    listofDirectories = os.listdir(inputDirectoryName)
    nFiles = len(listofDirectories)
    print(listofDirectories)
    print("\nthere are " + str(nFiles) + " text files in the \'" + inputDirectoryName + "\' directory:")

    # ignore ".DS_Store" directory
    if (".DS_Store" in listofDirectories):
        print("\".DS_Store\" element removed!")
        del listofDirectories[listofDirectories.index(".DS_Store")]

    nFiles = len(listofDirectories)
    print("\nthere are " + str(nFiles) + " text files in the \'" + inputDirectoryName + "\' directory:")
    print(listofDirectories)

    if (nFiles == 1):
        aDocName = listofDirectories[0]
        print("aDocName=" + aDocName)
        createGensimObjects(inputDirectoryName + '/' + aDocName)
    else:

        oneLargeOutputFile = outputDirectoryName + "/" + "mda_files.txt"
        print("\noneLargeOutputFile:" + oneLargeOutputFile)

        with open(oneLargeOutputFile, 'w') as outfile:
            for fname in listofDirectories:
                print("\nfname:" + fname)
                with open(inputDirectoryName + '/' + fname) as infile:
                    print("successfully wrote this file to mda_files.txt")
                    outfile.write(infile.read())

        createGensimObjects(oneLargeOutputFile)

    # end timer for the topic modelling process
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))

    sys.exit(0)
