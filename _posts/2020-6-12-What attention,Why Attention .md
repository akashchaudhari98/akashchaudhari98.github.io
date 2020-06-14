

---

## Introduction - 

Imagine you are at a super-market and looking to buy some cereals, how do you go about it ? do you look at all the items in the store all at once? no, you don't, you find the shelf for cereals and look only at them 
while ignoring the rest of the items in the store in other words you pay attention to the cereals shelf 
Attention networks do something similar with the help of deep learning in NLP 

the concept was introduced in the [paper](https://arxiv.org/pdf/1409.0473.pdf)  for neural machine translation, before looking at it in detail lets take a short look at what was used before this 

### Encoder-decoder architechture 

NMT was originally based on the seq2seq encoder-decoder architecture (introduced in this paper), this architecture had 3 important parts, the encoder, the context vector, and the decoder 
the encoder is an RNN( or an LSTM/RNN) that takes in the input sequence and converts it into a context vector 
the context vector is passed on to the decoder 
the decoder decodes this context vector to give an output sequence

![Image](https://smerity.com/media/images/articles/2016/gnmt_arch_1_enc_dec.svg)

#### Encoder 

Stack of several RNN(or LSTM/GRU for better performance) accepts single elements from the input and the previous hidden state, collects information from the input and passes on the hidden state to the next lstm 
the hidden states are computed as follows 

    
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;h_t&space;=&space;f(W_h_h.h_t_{-1},W_h_x.x_t)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;h_t&space;=&space;f(W_h_h.h_t_{-1},W_h_x.x_t)" title="\large h_t = f(W_h_h.h_t_{-1},W_h_x.x_t)" /></a>                               
                       
   W_hh are the weights associate with the previous input state <br>
   W_hx are the weights associated with the present input sequence


#### Decoder 

Stack of several RNN(or LSTM/GRU) cells, it accepts the last hidden state of the encoder as the context vector and cell state of the last encoder cell as the initial values and predicts the output sequence 

        
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;y_t&space;=&space;argmax(softmax(G(LSTM(e))))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;y_t&space;=&space;argmax(softmax(G(LSTM(e))))" title="\large y_t = argmax(softmax(G(LSTM(e))))" /></a>


### Drawbacks of seq2seq model 

1) The encoder-decoder network needs to compress all the information from the source sentence into a single foxed length vector this can create a problem in long sentences and sentences that are bigger than the sentences in the training corpus 

2) It does not take into account the amount of individual contribution of a word into consideration, attention, on the other hand, understands which words to focus during an individual time-step


## Attention mechanism 

![image](https://miro.medium.com/max/1332/0*VrRTrruwf2BtW4t5.)

(bahadanu attn)

Aiming to resolve the above issues the attention mechanism was introduced in the [paper](https://arxiv.org/abs/1409.3215), it maintains the same RNN encoder, but for each time-step, it computes an attention score for the hidden representations of each token.
Let the inputs be x and outputs be y 
		
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;x&space;=&space;[x_1&space;,&space;x_2&space;,...,x_n]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;x&space;=&space;[x_1&space;,&space;x_2&space;,...,x_n]" title="\large x = [x_1 , x_2 ,...,x_n]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;y&space;=&space;[y_1&space;,&space;y_2&space;,...,y_n]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;y&space;=&space;[y_1&space;,&space;y_2&space;,...,y_n]" title="\large y = [y_1 , y_2 ,...,y_n]" /></a>

We will be usinng bi-directional encoder , which read the sentence left to right as well as right to left , this to to include both the preceding as well as following words in annotation of one word
		 
forward hidden state = <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;\overrightarrow{h_i}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;\overrightarrow{h_i}" title="\large \overrightarrow{h_i}" /></a>

backward hidden state = <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;\overleftarrow{h_i}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;\overleftarrow{h_i}" title="\large \overleftarrow{h_i}" /></a>

source hidden state = <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;h_i&space;=&space;[\overrightarrow{h_i}:\overleftarrow{h_i}],&space;\hspace{1cm}&space;i&space;=&space;[1,2,...n]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;h_i&space;=&space;[\overrightarrow{h_i}:\overleftarrow{h_i}],&space;\hspace{1cm}&space;i&space;=&space;[1,2,...n]" title="\large h_i = [\overrightarrow{h_i}:\overleftarrow{h_i}], \hspace{1cm} i = [1,2,...n]" /></a>



In this model, conditional probability is defined as follows 


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;P(y_t|y_1.....y_t_{-1}&space;,&space;x)&space;=&space;g(y_t_{-1},s_t,c_t)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;P(y_t|y_1.....y_t_{-1}&space;,&space;x)&space;=&space;g(y_t_{-1},s_t,c_t)" title="\large P(y_t|y_1.....y_t_{-1} , x) = g(y_t_{-1},s_t,c_t)" /></a>

here g is a fully connected layer with non-linear activation and takes all the shows inputs as contactanated, s_i is the 
decoder hidden state for time-step i and is computed as 

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;s_t&space;=&space;f(s_t_{-1},y_t_{-1},c_t)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;s_t&space;=&space;f(s_t_{-1},y_t_{-1},c_t)" title="\large s_t = f(s_t_{-1},y_t_{-1},c_t)" /></a>               (f is an RNN/LSTM function)

c_i is the context vector which is computed for each time step using attention scores, which are calculated using an alignment model to score how well inputs at position i and outputs at position j match 
 

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;e_i_t&space;=&space;a(s_t_{-1},h_i)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;e_i_t&space;=&space;a(s_t_{-1},h_i)" title="\large e_i_t = a(s_t_{-1},h_i)" /></a>


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;\alpha_i_t&space;=&space;\frac{e_i_t}{\sum_{i&space;=1}^{n}(e_i_t)}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;\alpha_i_t&space;=&space;\frac{e_i_t}{\sum_{i&space;=1}^{n}(e_i_t)}" title="\large \alpha_i_t = \frac{e_i_t}{\sum_{i =1}^{n}(e_i_t)}" /></a>          

 (softmax for score normalisation) 


In Bahdanau’s paper, the alignment score  a() is parametrized by a feed-forward network, with a single hidden layer and this network is jointly trained with other parts of the model	
		

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;a&space;=&space;V_a.tanh(W_a[h_i:s_t])" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;a&space;=&space;V_a.tanh(W_a[h_i:s_t])" title="\large a = V_a.tanh(W_a[h_i:s_t])" /></a>

(fully connected layer , inputs are concatanated hidden state of bidirection encoder and 																							last hidden state of decoder)


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;c_t&space;=&space;\sum&space;(\alpha&space;_i_t.h_i)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;c_t&space;=&space;\sum&space;(\alpha&space;_i_t.h_i)" title="\large c_t = \sum (\alpha _i_t.h_i)" /></a>


c_t (context vector ) is a weighted average of the elements in the source sentence, it denotes the sentence representation concerning the current element h_j(out) and the similarity score e_ij, context vector is then combined with the current hidden state and the last target token y_t-1 to generate the current token y_j


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;y_t&space;=&space;f(s_t_{-1}&space;,y_t_{-1}&space;,&space;c_t)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;y_t&space;=&space;f(s_t_{-1}&space;,y_t_{-1}&space;,&space;c_t)" title="\large y_t = f(s_t_{-1} ,y_t_{-1} , c_t)" /></a>



<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;s_t_{-1}&space;=&space;f(h_t_{-1},y_t_{-1})" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;s_t_{-1}&space;=&space;f(h_t_{-1},y_t_{-1})" title="\large s_t_{-1} = f(h_t_{-1},y_t_{-1})" /></a>


This procedure is repeated for each token y_t until the end of the output sequence 

Attention score can be calculated in multiple ways, here are some popular attention score mechanisms 


## Global attention 
![image](https://miro.medium.com/max/602/1*WmXfpYa_aIOMYTavmFW4aw.png)

The idea of a global attention model is to consider all hidden states of encoder while deriving the context vector c_t, here the score is determined by comparing the current hidden state with each source hidden state


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;e_i_t&space;=&space;score(h_t,\bar{h}_s)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;e_i_t&space;=&space;score(h_t,\bar{h}_s)" title="\large e_i_t = score(h_t,\bar{h}_s)" /></a>


		
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;\alpha&space;_i_t&space;=\frac{e_i_t}{\sum&space;(e_i_t)}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;\alpha&space;_i_t&space;=\frac{e_i_t}{\sum&space;(e_i_t)}" title="\large \alpha _i_t =\frac{e_i_t}{\sum (e_i_t)}" /></a>
		
(softmax for score normalisation)

there are three ways of calculating scores 

	  		  

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;score&space;=&space;h_t^T.\bar{h}_s" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;score&space;=&space;h_t^T.\bar{h}_s" title="\large score = h_t^T.\bar{h}_s" /></a>              (dot)


	 		   
	
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;score&space;=&space;h_t^T.W_a.\bar{h}_s" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;score&space;=&space;h_t^T.W_a.\bar{h}_s" title="\large score = h_t^T.W_a.\bar{h}_s" /></a>    (general)


	

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;score&space;=&space;V_a.tanh(W_a[h_t:h_s])" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;score&space;=&space;V_a.tanh(W_a[h_t:h_s])" title="\large score = V_a.tanh(W_a[h_t:h_s])" /></a>  	 (concat)

In spirit global attention is similar to bahdanau attention which we discussed before but there are certain differences between the two, while bahadanu using a concatenation of the forward and backward encoder stats with the previous hidden state of the target sequence, global attention simply uses the top hidden state of the encoder and decoder LSTM.


## Local Attention 

![image](https://miro.medium.com/max/600/1*rcHVQN4QGwNWBOs69QeJsg.png)


It selectively focuses on a small window of context and is differentiable, first the model generates an "aligned position" p_t for each target word at time t, the context vector c_t is then derived as a weighted average over a set of encoder hidden states within the window (p_t - D, p_t + D) here D is empirically selected, if the window crosses sentence boundaries then we simply ignore the outside part and focus on the words inside the window

local attention model has two variants 

*monotonic alignment* - here the position vector p_t is set to p_t = t, assuming the source sentence and target sentence are monotonically aligned, the alignment is calculated as 


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;\alpha&space;_i_t&space;=&space;align(h_t,\bar{h}_t)&space;=&space;softmax(score(h_t,\bar{h}_t))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;\alpha&space;_i_t&space;=&space;align(h_t,\bar{h}_t)&space;=&space;softmax(score(h_t,\bar{h}_t))" title="\large \alpha _i_t = align(h_t,\bar{h}_t) = softmax(score(h_t,\bar{h}_t))" /></a>

monotonic alignment is nearly the same as global attention, except that vector α_it is of fixed length and shorter

*predictive alignment* - unlike monotonic alignment here we do not assume the value of p_t we predict using the following equation


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;p_t&space;=&space;S.sigmoid(v_p^T.tanh(W_p.h_t))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;p_t&space;=&space;S.sigmoid(v_p^T.tanh(W_p.h_t))" title="\large p_t = S.sigmoid(v_p^T.tanh(W_p.h_t))" /></a>

W_p and v_p are model parameters that are trained along with the rest of the model 
S is the length of the source sequence
since we have added sigmoid the range of p_t is [0, S]

the alignment is calculated as follows 


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;\alpha&space;_i_t&space;=&space;align(h_t,\bar{h}_t).e^{\frac{-(s-p_t)^2}{2sd^2}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;\alpha&space;_i_t&space;=&space;align(h_t,\bar{h}_t).e^{\frac{-(s-p_t)^2}{2sd^2}}" title="\large \alpha _i_t = align(h_t,\bar{h}_t).e^{\frac{-(s-p_t)^2}{2sd^2}}" /></a>

to favor points near p_t a Gaussian distribution is centered around p_t 


## Hierarchical Attention

![image](https://humboldt-wi.github.io/blog/img/seminar/group5_HAN/han_architecture.jpg)

This model is mostly used for applications such as document classification, it has a hierarchical structure which mirrors
the Hierarchical structure of the model has two levels of attention mechanisms that are applied at word level and sentence level enabling it to more and less important content when constructing document representation. 
If we think of a document it has a nested structure which is as follows 

character > word > sentence > document

a hierarchical structure is constructed accordingly either from the doc to char ( top-bottom) or the other way round (bottom-up).
The network constructs the document representation by building representations of sentences and then aggregating those into a document representation.
the sentence representations are built by encoding the words in the sentence and applying attention mechanism on them resulting in a sentence representation the document representation is built in the same manner, but it only receives the sentence vector as input.

At word level we use bidirectional GRU as RNN cells, this gives us word annotations which summarizes information form the backward as well as forward direction resulting in a variable h_it 


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;h_i_t&space;=&space;[\overrightarrow{h_i_t}&space;;\overleftarrow{h_i_t}]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;h_i_t&space;=&space;[\overrightarrow{h_i_t}&space;;\overleftarrow{h_i_t}]" title="\large h_i_t = [\overrightarrow{h_i_t} ;\overleftarrow{h_i_t}]" /></a>


We then apply the attention mechanism as we did in before 
		

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;e_i_t&space;=&space;score(h_i_t)=&space;tanh(W_w.h_i_t&space;&plus;&space;b_w)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;e_i_t&space;=&space;score(h_i_t)=&space;tanh(W_w.h_i_t&space;&plus;&space;b_w)" title="\large e_i_t = score(h_i_t)= tanh(W_w.h_i_t + b_w)" /></a>

we use tanh to keep the values between [-1 , 1]


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;\alpha&space;_i_t&space;=&space;softmax(tanh(W_w.h_i_t&space;&plus;&space;b_w))&space;=&space;e^{\frac{tanh(W_w.h_it&space;&plus;&space;b_w))}{\sum&space;e^{tanh(W_w.h_it&space;&plus;&space;b_w}}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;\alpha&space;_i_t&space;=&space;softmax(tanh(W_w.h_i_t&space;&plus;&space;b_w))&space;=&space;e^{\frac{tanh(W_w.h_it&space;&plus;&space;b_w))}{\sum&space;e^{tanh(W_w.h_it&space;&plus;&space;b_w}}}" title="\large \alpha _i_t = softmax(tanh(W_w.h_i_t + b_w)) = e^{\frac{tanh(W_w.h_it + b_w))}{\sum e^{tanh(W_w.h_it + b_w}}}" /></a>



<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;s_i&space;=&space;\sum&space;(\alpha&space;_i_t.h_i_t)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;s_i&space;=&space;\sum&space;(\alpha&space;_i_t.h_i_t)" title="\large s_i = \sum (\alpha _i_t.h_i_t)" /></a>

At the sentence level, we repeat the same procedure but with s_i as input to the bidirectional GRU cells


<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\large&space;v&space;=&space;\sum&space;(\alpha&space;_i.h_i)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\fn_phv&space;\large&space;v&space;=&space;\sum&space;(\alpha&space;_i.h_i)" title="\large v = \sum (\alpha _i.h_i)" /></a>

Trainable weights and biases are randomly initialized and jointly learned during the training process.
The final output is a document vector v which can be used as features for document classification.
	

## Refrences

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)<br>
[Attention ? Attention !](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#definition)<br>
[An Introductory Survey on Attention Mechanisms in NLP Problems](https://arxiv.org/abs/1811.05544)<br>
[NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)<br>
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)<br>
[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)<br>
[Text Classification with Hierarchical Attention Networks](https://humboldt-wi.github.io/blog/research/information_systems_1819/group5_han/)<br>
