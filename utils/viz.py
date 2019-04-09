from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def sims_aligned(year, word, *args):
    
    all_most_sim = []
    wrd_vectors = []
    model = args[0]
    

    def unite_sims(m):
        for x in m:
            all_most_sim.append(x[0])
    
    unite_sims(model.most_similar([word], topn=7))
    
    for i in range(1, len(args)):
        m1,m2 = intersection_align_gensim(args[0], args[i])
        unite_sims(m2.most_similar([word], topn=7))
        wrd_vectors.append([m2[word]])
        
    return all_most_sim, wrd_vectors, model, word, year

    
def viz(all_most_sim, wrd_vectors, model, word, year):
        
    arr = np.empty((0,300), dtype='f')
    word_labels = [word+' '+year]

    num = int(year) - 1
    
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for i in range(0, len(wrd_vectors)):
        arr = np.append(arr, np.array(wrd_vectors[i]), axis=0)
        word_labels.append(word+' '+str(num))
        num -= 1
        
    
    for i in all_most_sim:
        wrd_vector = model[i]
        word_labels.append(i)
        arr = np.append(arr, np.array([wrd_vector]), axis=0)


    tsne = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca')
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    
    wrd_coord = []
    
    for i in range(0, len(wrd_vectors)+1):
        a = Y[i]
        wrd_coord.append(a)
    

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    plt.scatter(x_coords, y_coords)
    plt.axis('off')

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()-10, x_coords.max()+10)
    plt.ylim(y_coords.min()-10, y_coords.max()+10)
    for i in range(len(wrd_coord)-1,0,-1):
        plt.annotate("", xy = (wrd_coord[i-1][0], wrd_coord[i-1][1]), xytext = (wrd_coord[i][0], wrd_coord[i][1]), arrowprops=dict(arrowstyle='-|>', color='indianred'))
   
    plt.show()
    
    
def main():
    all_most_sim, wrd_vectors, model, word, year = sims_aligned('2002', 'железный_ADJ', model_2002, model_2001, model_2000)
    viz(all_most_sim, wrd_vectors, model, word, year)
    
    
if __name__ == '__main__':
    main()

