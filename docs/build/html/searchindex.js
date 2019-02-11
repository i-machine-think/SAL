Search.setIndex({docnames:["dataset","evaluator","index","loss","models","notes/intro","optim","trainer","util"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":1,"sphinx.ext.viewcode":1,sphinx:54},filenames:["dataset.rst","evaluator.rst","index.rst","loss.rst","models.rst","notes/intro.md","optim.rst","trainer.rst","util.rst"],objects:{"machine.dataset":{fields:[0,0,0,"-"]},"machine.dataset.fields":{SourceField:[0,1,1,""],TargetField:[0,1,1,""]},"machine.dataset.fields.SourceField":{build_vocab:[0,2,1,""]},"machine.dataset.fields.TargetField":{SYM_EOS:[0,3,1,""],SYM_SOS:[0,3,1,""],build_vocab:[0,2,1,""],include_eos:[0,3,1,""]},"machine.evaluator":{evaluator:[1,0,0,"-"],predictor:[1,0,0,"-"]},"machine.evaluator.evaluator":{Evaluator:[1,1,1,""]},"machine.evaluator.evaluator.Evaluator":{compute_batch_loss:[1,2,1,""],evaluate:[1,2,1,""],update_batch_metrics:[1,4,1,""],update_loss:[1,4,1,""]},"machine.evaluator.predictor":{Predictor:[1,1,1,""]},"machine.evaluator.predictor.Predictor":{predict:[1,2,1,""]},"machine.loss":{loss:[3,0,0,"-"]},"machine.models":{DecoderRNN:[4,0,0,"-"],EncoderRNN:[4,0,0,"-"],TopKDecoder:[4,0,0,"-"],attention:[4,0,0,"-"],baseRNN:[4,0,0,"-"],seq2seq:[4,0,0,"-"]},"machine.models.DecoderRNN":{DecoderRNN:[4,1,1,""]},"machine.models.DecoderRNN.DecoderRNN":{forward:[4,2,1,""],forward_step:[4,2,1,""]},"machine.models.EncoderRNN":{EncoderRNN:[4,1,1,""]},"machine.models.EncoderRNN.EncoderRNN":{forward:[4,2,1,""]},"machine.models.TopKDecoder":{TopKDecoder:[4,1,1,""]},"machine.models.TopKDecoder.TopKDecoder":{forward:[4,2,1,""]},"machine.models.attention":{Attention:[4,1,1,""],Concat:[4,1,1,""],Dot:[4,1,1,""],MLP:[4,1,1,""]},"machine.models.attention.Attention":{forward:[4,2,1,""],get_method:[4,2,1,""],set_mask:[4,2,1,""]},"machine.models.attention.Concat":{forward:[4,2,1,""]},"machine.models.attention.Dot":{forward:[4,2,1,""]},"machine.models.attention.MLP":{forward:[4,2,1,""]},"machine.models.baseRNN":{BaseRNN:[4,1,1,""]},"machine.models.baseRNN.BaseRNN":{forward:[4,2,1,""]},"machine.models.seq2seq":{Seq2seq:[4,1,1,""]},"machine.models.seq2seq.Seq2seq":{flatten_parameters:[4,2,1,""],forward:[4,2,1,""]},"machine.optim":{optim:[6,0,0,"-"]},"machine.optim.optim":{Optimizer:[6,1,1,""]},"machine.optim.optim.Optimizer":{set_scheduler:[6,2,1,""],step:[6,2,1,""],update:[6,2,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:staticmethod"},terms:{"class":[0,1,4,6],"default":[1,4,6],"float":[1,4,6],"function":[4,5,6],"int":[4,6],"new":5,"return":[1,4],"static":1,"true":[0,4],"try":5,"while":4,EOS:5,For:[0,5],IDs:4,The:[4,5,6],There:5,Use:5,Used:4,about:0,accuraci:[1,5],activ:4,addit:[4,5],after:5,afterward:4,against:1,align:4,all:[0,4,5],allow:4,also:0,although:4,anh:4,ani:5,append:0,appli:4,arbitrari:4,architectur:4,arg:[0,4],argument:[0,4],arrai:4,attend:4,attent:5,attention_method:4,attention_method_kwarg:4,attn:4,avail:5,base:4,batch:[1,4,5],batch_first:0,batch_siz:4,beam:4,becom:4,befor:5,below:5,bidirect:4,bidirection:5,bool:4,both:5,build_vocab:0,c_0:4,call:4,caller:6,can:[0,5],care:4,cell:4,chang:5,checkpoint_path:5,clip:6,column:0,com:[0,5],command:5,commandlin:5,commit:5,complet:5,compon:4,comput:[1,4],compute_batch_loss:1,concat:4,concaten:4,configur:4,construct:0,constructor:0,contain:[4,5],context:4,contribut:2,correspond:0,could:6,criteria:6,ctrl:5,current:[1,5,6],data:[0,1,5],data_iter:1,dataset:[1,2,5],decod:[1,4],decode_funct:4,decoder_hidden:[1,4],decoder_output:[1,4],decoder_rnn:4,decoder_st:4,defin:4,depend:6,detail:4,dev:5,dev_path:5,develop:[5,6],dict:1,dictionari:4,differ:1,dim:4,dimens:4,directli:[0,4],directori:5,disabl:6,discov:5,distribut:4,diverg:5,dot:4,drawn:4,dropout:4,dropout_p:4,dure:5,each:4,egin:4,element:4,embed:[4,5],embedding_s:[4,5],encapsul:6,encod:4,encoder_hidden:4,encoder_output:4,encoder_st:4,end:[0,4],enter:5,environ:5,eos:0,eos_id:[0,4],epoch:[5,6],evalu:2,everi:4,exampl:4,exist:5,exp:4,expect:[4,5],expt_dir:5,extra:1,fals:4,featur:4,flag:4,flatten:4,flatten_paramet:4,folder:5,follow:4,forc:[0,4,5],fork:5,former:4,forward:[1,4],forward_rnn:4,forward_step:4,framework:4,from:[0,4,5],full:4,full_focu:4,gener:[1,4,5],get_batch_data:1,get_method:4,github:[0,5],given:[1,4,6],gradient:6,gru:[4,5],h_0:4,has:5,heavi:5,help:5,hidden:[1,4,5],hidden_s:[4,5],hook:4,http:[0,5],ibm:5,ignor:4,illustr:5,implement:[4,5],includ:[5,6],include_eo:0,include_length:0,index:[0,4],indic:4,individu:0,inf:4,inform:[0,4],initi:4,input:[1,4,5],input_dropout_p:4,input_len:4,input_length:4,input_s:4,input_var:4,input_vocab:4,instal:5,instanc:4,instanti:6,instead:4,integ:4,integr:5,integration_test:5,introduct:2,invers:5,iter:[0,1],its:5,kei:[1,4],key_attn_scor:4,key_input:4,key_length:4,key_sequ:4,keyword:[0,4],kwarg:[0,4],languag:1,last:4,latter:4,layer:[4,5],learn:[5,6],length:4,librari:5,line:5,list:[1,4],load:[0,5],log_softmax:4,look:4,loss:[1,2,5,6],lr_schedul:6,lstm:4,machin:[0,1,5,6],make:1,manag:0,map:1,mask:4,max_grad_norm:6,max_len:4,max_length:4,max_seq_length:4,maximum:4,mechan:4,met:6,method:4,metric:1,mini:4,mlp:4,mode:5,model:[1,5],modul:4,more:0,multi:4,multipl:4,must:4,n_layer:4,name:5,necessari:6,need:4,nllloss:1,none:4,norm:6,note:[2,4],num_direct:4,num_lay:4,number:[4,5,6],numpi:5,object:[0,1,4,6],onc:5,one:[0,4],optim:[2,5],option:[1,4,5,6],origin:5,other:[0,1],out:5,output:[1,4,5],output_dir:5,output_len:4,over:4,overridden:4,overview:5,packag:[2,6],param:6,paramet:[0,1,4,6],pass:[0,1,4],perform:[1,4,6],pip:5,pleas:[0,5],posit:0,possibl:0,pre:1,predict:[1,4,5],predicted_softmax:4,prepend:0,preprocess:0,previou:4,print:5,probabl:4,process:[0,4],project:5,prompt:5,provid:[0,4,6],pull:5,python:5,pytorch:[0,5],quickstart:2,randn:4,random:4,rate:6,recip:4,recurr:4,refer:[2,5],regist:4,remain:0,repositori:5,repres:[0,4],request:5,requir:2,respect:5,resum:5,ret_dict:4,retain_output_prob:4,retrain:5,revers:5,right:5,rnn:4,rnn_cell:[4,5],run:[4,5],sampl:4,schedul:6,scratch:5,search:4,sentenc:[0,4],seq2seq:[4,5],seq_len:4,sequenc:[0,4,5],sequenceaccuraci:1,set:[0,4,5,6],set_mask:4,set_schedul:6,sever:5,sgd:6,shape:4,should:[4,6],silent:4,simpl:5,sinc:4,singl:6,size:[4,5],smaller:4,softmax:4,sos:0,sos_id:[0,4],sourc:[0,1,4,5,6],sourcefield:0,specifi:4,src_seq:1,src_vocab:1,standard:4,start:[0,4],state:[1,4,5],step:[0,4,6],steplr:6,store:5,str:4,sub:4,subclass:4,substanti:5,sum_j:4,supervis:6,sym_eo:[0,4],sym_mask:4,sym_so:0,symbol:[0,4],take:4,target:[1,4,5],target_vari:1,targetfield:0,teacher:[4,5],teacher_forcing_ratio:4,tensor:[1,4],termin:5,test:5,test_data:5,text:0,tgt_seq:1,tgt_vocab:1,than:4,them:4,thi:[0,4,5],think:5,three:5,time:4,toi:5,token:[1,4],tool:5,toolkit:5,top:[4,5],topk_length:4,topk_sequ:4,torch:[1,4,6],torchtext:[0,1,5],train:[1,6],train_model:5,train_path:5,trainer:[2,5,6],translat:5,tupl:4,type:[1,4],uniformli:4,unittest:5,updat:[1,6],update_batch_metr:1,update_loss:1,usag:5,use:[0,4,5],use_attent:4,used:[0,4,5,6],uses:[5,6],using:[4,5],util:2,valu:[0,4,6],variabl:[0,4],variable_length:4,vector:4,version:5,visit:5,vocab:0,vocab_s:4,vocabulari:[4,5],websit:5,weight:4,welcom:5,when:[4,6],where:4,whether:4,which:[0,4,5],whose:4,within:4,wordaccuraci:1,would:4,wrapper:0,x_i:4,x_j:4,you:5,your:5,zero:4},titles:["Dataset","Evaluator","Codebase for the i-machine-think project, a modular and fully tested Pytorch implementation for seq2seq models","Loss","Models","Introduction","Optim","Trainer","Util"],titleterms:{attent:4,basernn:4,checkpoint:[5,8],codebas:2,contribut:5,dataset:0,decoderrnn:4,encoderrnn:4,evalu:[1,5],exampl:5,field:0,fulli:2,implement:2,infer:5,introduct:5,loss:3,machin:[2,4],model:[2,4],modular:2,nllloss:3,optim:6,perplex:3,predictor:1,project:2,pytorch:2,quickstart:5,requir:5,script:5,seq2seq:2,supervised_train:7,test:2,think:2,topkdecod:4,train:5,trainer:7,util:8}})