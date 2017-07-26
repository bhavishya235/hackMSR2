How to run this?

Tested on linux or mac machine:

1. Install NLTK	<sudo pip install nltk>

2. Download stanford coreNLP from https://stanfordnlp.github.io/CoreNLP/download.html and extract it at say path <PATH_STAN_NLP>
	
3. Install stanford-corenlp	<pip install stanfordcorenlp>	

4. Install flask. <pip install flask-restful>

6. Open a terminal window, go to PATH_STAN_NLP directory and run following command:
	<java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000>

	This will run the stanford NLP service on localhost:9000. You can also mention hostname using -host i.e.
	<java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -host 127.99.99.99 -port 9000 -timeout 15000>

7. Specify this host and port in file main-api.py in variables “stan_nlp_host” and “stan_nlp_port” respectively

8. Specify “flask_host” with some public IP as well in main-api.py. This IP will be used as an API address

How to use API?

RestAPI: http://<flask_host>/api/Victim_Classifier?text=<current comment>&prevOffScore=<offender classifier score of prev comment>&currOffScore=<offender classifier score of current comment>

Return: Single string: “not_victim” or “maybe” or “likely”  or “confident” or “very_confident”
As we go right degree of being victim increases