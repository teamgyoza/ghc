ghc
===

ghc2014



Gcutil
=======

https://developers.google.com/compute/docs/gcutil/#install
project-id arboreal-cat-537

gcutil addinstance 



After install
================================
sudo apt-get install vim git

gcutil ssh <nom de la machine>
	
	ou

ssh -o UserKnownHostsFile=/dev/null -o CheckHostIP=no -o StrictHostKeyChecking=no -i $HOME/.ssh/google_compute_engine -A -p 22 $USER@TYPE-GOOGLE-COMPUTE-ENGINE-PUBLIC-IP-HERE

e.g.
ssh -o UserKnownHostsFile=/dev/null -o CheckHostIP=no -o StrictHostKeyChecking=no -i $HOME/.ssh/google_compute_engine -A -p 22 paul@23.251.136.73



sshfs
===============================

sshfs paul@23.251.136.73:/home/paul /media/gyoza/ -o UserKnownHostsFile=/dev/null -o CheckHostIP=no -o StrictHostKeyChecking=no


Conda
======
bzip2 requis
wget http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-1.9.1-Linux-x86_64.sh
Environnement python avec scikit numpy etc, precompile
http://continuum.io/downloads





HADOOP

 ./compute_cluster_for_hadoop.py setup arboreal-cat-537  hdoop-test
 
./compute_cluster_for_hadoop.py start arboreal-cat-537 hdoop-test 4 --data-disk-gb 3

./compute_cluster_for_hadoop.py mapreduce arboreal-cat-537 hdoop-test --input gs://janeausteen --output  gs://janeausteenoutput --mapper sample/shortest-to-longest-mapper.py  --reducer sample/shortest-to-longest-reducer.py --mapper-count 3 --reducer-count 1
