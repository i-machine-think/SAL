import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        self.model = model.to(device)

        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq, tgt_seq=None, attn=None):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
# <<<<<<< HEAD
#         target_variable = None
#         src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
#                               volatile=True).view(1, -1)
#         if (tgt_seq != None):
#             tgt_id_seq = Variable(torch.LongTensor([self.tgt_vocab.stoi[tok] for tok in tgt_seq]),
#                                   volatile=True).view(1, -1)
#
#             attn_target = Variable(torch.LongTensor(attn[:-1]).unsqueeze(0))
#
#             if torch.cuda.is_available():
#                 tgt_id_seq = tgt_id_seq.cuda()
#                 attn_target = attn_target.cuda()
#             target_variable = {'decoder_output': tgt_id_seq, 'attention_target': attn_target}
#
#
#         if torch.cuda.is_available():
#             src_id_seq = src_id_seq.cuda()
# =======
        src_id_seq = torch.tensor([self.src_vocab.stoi[tok] for tok in src_seq], dtype=torch.long, device=device).view(1, -1)




        softmax_list, _, other = self.model(src_id_seq, [len(src_seq)], target_variable)
        length = other['length'][0]
        # print("length:{}".format(other['length'][0]))
        # input()
        try:
            attn_vec_len = other['attention_score'][0].data.size()[-1]
            attention_tensors = [other['attention_score'][di][0].data for di in range(length)]

            attentions = torch.zeros(length, attn_vec_len)
            for i in range(length):
                attentions[i,:] = attention_tensors[i]

            attentions = attentions.numpy()
        except:
            attentions = 0
            print("attention switched off")
        # print("attentions:{}".format(attentions.shape))
        # input()

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq, attentions
