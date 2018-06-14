import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

class PlotAttention(object):

    def __init__(self, master_data):
        self.master_data = master_data

    def _showAttention(self,input_sentence, output_words, attentions,name,colour):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.yaxis.tick_right()
        cax = ax.matshow(attentions, cmap='bone', vmin=0, vmax=1)
        #fig.colorbar(cax)
        cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.8])
        cb = plt.colorbar(cax, cax=cbaxes)
        cbaxes.yaxis.set_ticks_position('left')

        # Set up axes
        ax.set_xticks(np.arange(len(input_sentence.split()) + 1))
        ax.set_yticks(np.arange(len(output_words) + 1))
        ax.set_xticklabels([''] + input_sentence.split(' '), rotation=0) #+['<EOS>']
        ax.set_yticklabels([''] + output_words)

        #Colour ticks
        for ytick, color in zip(ax.get_yticklabels()[1:], colour):
            ytick.set_color(color)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        #X and Y labels
        ax.set_xlabel("INPUT")
        ax.set_ylabel("OUTPUT")
        ax.yaxis.set_label_position('right')
        ax.xaxis.set_label_position('top')

        plt.savefig("{}.png".format(name))
        plt.close(fig)
        #plt.show()


    def evaluateAndShowAttention(self, input_sentence, output_words, attentions,name=None):
        ipt = input_sentence.split(' ')
        nis = ipt[0]
        tgt = ipt[0]
        i = 1
        colour = []
        while(i<=len(ipt)):
            if(output_words[i-1]==tgt):
                colour.append('g')
            else:
                colour.append('r')
            if(len(output_words) < len(ipt)):
                if(i==len(output_words)):
                    break

            elif(i==len(ipt)):
                break

            temp = tgt + ' ' + ipt[i]
            row = np.where(self.master_data[:, 0] == temp)[0]
            tgt = self.master_data[row, 1][0].split(' ')[1]
            nis += ' ' +ipt[i] + '({})'.format(tgt)
            i+=1
        #row_t = np.where(self.master_data[:,0]==input_sentence)[0]
        #target = self.master_data[row_t,1]
        # print('input =', input_sentence)
        # print('target =', target)
        # print('output =', ' '.join(output_words))
        self._showAttention(input_sentence, output_words, attentions,name,colour)