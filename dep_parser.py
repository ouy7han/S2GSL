import copy

class DepInstanceParser():
    def __init__(self,basicDependencies,tokens=[]):
        self.basicDependencies=basicDependencies
        self.tokens=tokens
        self.words=[]
        self.dep_governed_info=[]
        self.dep_parsing()


    def dep_parsing(self):
        if len(self.tokens)>0:
            words=[]
            for token in self.tokens:
                words.append(token)
            dep_governed_info=[
                {"word":word}
                for i,word in enumerate(words)
            ]
            self.words=words
        else:
            dep_governed_info=[{}]*len(self.basicDependencies)
        for dep in self.basicDependencies:
            dependent_index=dep['dependent']-1
            governed_index=dep['governor']-1
            dep_governed_info[dependent_index]={
                "governor":governed_index,
                "dep":dep['dep']
            }
        self.dep_governed_info=dep_governed_info
        #print(dep_governed_info)





    def get_init_dep_matrix(self):
        dep_type_matrix=[["none"]*len(self.words) for _ in range(len(self.words))]
        for i in range(len(self.words)):
            dep_type_matrix[i][i]="self_loop"
        return dep_type_matrix

    def get_first_order(self,direct=False):
        dep_type_matrix=self.get_init_dep_matrix()

        for i,dep_info in enumerate(self.dep_governed_info):
            governor=dep_info["governor"]
            dep_type=dep_info["dep"]
            dep_type_matrix[i][governor]=dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governor][i]=dep_type if direct is False else "{}_out".format((dep_type))

        return dep_type_matrix
