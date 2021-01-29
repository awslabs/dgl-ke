from .module import Module
from  refactor.nn.modules import KGEmbedding
import itertools

# this is an interface so it will not have many parameters to be defined.
class Encoder(Module):
    def __init__(self, encoder_name):
        super(Encoder, self).__init__(encoder_name)

    def encode(self, data):
        raise NotImplementedError

class KGEEncoder(Encoder):
    def __init__(self, hidden_dim, n_entity, n_relation, init_func: list, encoder_name='KGEEncoder'):
        """ Initialize a knowledge graph encoder.

        Parameters
        ----------
        encoder_name: str
            the name of encoder.
        hidden_dim: int
            the dimension for each entity/relation.
        n_entity: int
            how many entities are there in the knowledge graph.
        n_relation: int
            How many relations are there in the knowledge graph.
        init_func: list
            Each element in init_func is a partial function. It is passed into entity_emb and relation_emb
        to initialize the parameters of them.

        """
        super(KGEEncoder, self).__init__(encoder_name)
        self.entity_emb = KGEmbedding()
        self.relation_emb = KGEmbedding()
        self.entity_emb.init(num=n_entity, dim=hidden_dim, init_func=init_func[0])
        self.relation_emb.init(num=n_relation, dim=hidden_dim, init_func=init_func[1])

    def save(self, save_path):
        self.entity_emb.save(save_path, 'entity_emb.pth')
        self.relation_emb.save(save_path, 'relation_emb.pth')

    def load(self, load_path):
        self.entity_emb.load(load_path, 'entity_emb.pth')
        self.relation_emb.load(load_path, 'relation_emb.pth')

    def sparse_parameters(self):
        return list(itertools.chain(self.entity_emb.parameters(), self.relation_emb.parameters()))

    def train(self):
        self.entity_emb.train()
        self.relation_emb.train()

    def eval(self):
        self.entity_emb.eval()
        self.relation_emb.eval()

    def encode(self, data):
        # encoder must know the metadata of data
        if 'lcwa' in data.keys() and data['lcwa'] is True:
            head = self.entity_emb(data['head'])
            tail = self.entity_emb(data['tail'])
            rel = self.relation_emb(data['rel'])
            return [head, rel, tail]
        else:
            head = self.entity_emb(data['head'])
            tail = self.entity_emb(data['tail'])
            rel = self.relation_emb(data['rel'])
            negs = self.entity_emb(data['negs'])
            return [head, rel, tail, negs]

    def prepare_distributed_training(self, gpu_id=-1, rank=0, world_size=1):
        if world_size > 1:
            self.global_relation_emb = self.relation_emb
            self.relation_emb = self.relation_emb.to(gpu_id)
        else:
            self.entity_emb = self.entity_emb.to(gpu_id)
            self.relation_emb = self.relation_emb.to(gpu_id)


    def share_memory(self):
        self.relation_emb.share_memory()
        self.entity_emb.share_memory()