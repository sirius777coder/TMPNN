# tmpnn
Graph based deep learning method to design protein sequences.There are twon main models which are TMPNN-alpha and TMPNN-beta. 

1. TMPNN-alpha is only used for inverse folding (also called fixed backbone design).
2. TMPNN-beta v1.0 is trained to not only capture the residue identity information but the residue topology information. We add a conditional random field after the encoder block.
3. TMPNN-beta v2.0 add an IPA module from OpenFold to explicit model the backbone information.

----
Author : Bo Zhang

E-mail : zhangbo777@sjtu.edu.cn