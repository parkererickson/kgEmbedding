from spektral.data import Dataset
import pandas as pd
import numpy as np
import os
from spektral.data import Graph

class KgDataset(Dataset):
    def __init__(self, entitiesWithEmbeddingFname, entityRelationFname, relationFname, relationEmbFname, **kwargs):
        self.entEmbFname = entitiesWithEmbeddingFname
        self.entRelationsFname = entityRelationFname
        self.relationsFname = relationFname
        self.relEmbFname = relationEmbFname
        self.numGraphs = None

        super().__init__(**kwargs)

    def download(self):
        entitiesEmb = pd.read_csv(self.entEmbFname, sep="\t", index_col=0)
        def convEmbToNumpy(row):
            return np.fromstring(row["embedding"].strip("[ ").strip(" ]"), sep=" ")
        entitiesEmb["embedding"] = entitiesEmb.apply(convEmbToNumpy, axis=1)
        eEmbDict = pd.Series(entitiesEmb.embedding.values,index=entitiesEmb.entity).to_dict()
        entityRel = pd.read_csv(self.entRelationsFname, sep="\t", header=None)
        entityRel.rename(columns={0:"src", 1:"rel", 2:"dest"}, inplace=True)
        relations = pd.read_csv(self.relationsFname, sep="\t", header=None)
        relations.rename(columns={0:"idx", 1:"relationType"}, inplace=True)
        relEmb = np.load(self.relEmbFname)
        relations["emb"] = [relEmb[i] for i in relations["idx"]]
        rEmbDict = pd.Series(relations.emb.values,index=relations.relationType).to_dict()
        entityRel["srcEmb"] = [eEmbDict[i] for i in entityRel["src"]]
        entityRel["destEmb"] = [eEmbDict[i] for i in entityRel["dest"]]
        entityRel["relEmb"] = [rEmbDict[i] for i in entityRel["rel"]]
        def collectGraphs(row):
            vid = row["entity"]
            data = entityRel[entityRel["src"] == vid]
            feats = np.stack(data["destEmb"].values)
            y = data["srcEmb"].values[0]
            adj = np.ones((feats.shape[0], feats.shape[0]))
            filename = os.path.join(self.path, f'kg_graph_{row.name}')
            np.savez(filename, x=feats, a=adj, y=y)
        
        os.mkdir(self.path)
        pats = entitiesEmb.loc[entitiesEmb["type"] == "Patient"]
        pats.apply(collectGraphs, axis=1)
        self.numGraphs = entitiesEmb.shape[0]

    def read(self):
        # We must return a list of Graph objects
        output = []
        self.numGraphs = len([name for name in os.listdir(self.path)])
        print(self.numGraphs)
        print(self.path)
        for i in range(self.numGraphs):
            try:
                data = np.load(os.path.join(self.path, f'kg_graph_{i}.npz'))
                output.append(
                    Graph(x=data['x'], a=data['a'], y=data['y'])
                )
            except:
                pass

        return output