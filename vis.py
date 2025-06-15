import matplotlib.pyplot as plt                                   # type: ignore
import networkx as nx                                             # type: ignore
import numpy as np                                                # type: ignore
import textwrap
from nanograd import Variable, Parameter, squared_error, lasso, trace, Walk
from util import RandomSeed


# create dataset
K = 4    # number of features
N = 300  # number of examples
with RandomSeed(3) as rng:
    x = np.hstack((np.ones((N, 1)), rng.normal(size=(N, K))))
    v = rng.normal(0.0, 1.0, size=(K+1, 1))
    w = rng.normal(0.0, 1.0, size=(K+1, 1))
    e = rng.normal(0.0, 0.2, size=(N, 1))
    t = x @ v + e


# define model
t = Variable (t, tag='t')
x = Variable (x, tag='x')
w = Parameter(w, tag='w')
y = x @ w
loss = squared_error(y, t) + Variable(0.1, tag='λ') * lasso(w)


# trace graph
nodes: list[str] = list()
edges: list[tuple[str, str]] = list()
trace(loss, Walk(nodes, edges))
edges = sorted(edges, key=lambda uv: nodes.index(uv[1]))


# make graph
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
nx.write_network_text(G, ascii_only=True, vertical_chains=True)
labels = {
    n: '\n'.join(textwrap.wrap(str(n), width=25))
    for n in G.nodes()
}
pos = nx.kamada_kawai_layout(G)


# draw graph
nx.draw(G, pos, with_labels=False, arrows=True)
nx.draw_networkx_labels(
    G, pos, labels,
    font_family='DejaVu Sans Mono',
)
ax = plt.gca()
ax.margins(0.3)
# plt.savefig('comp-graph.png', dpi=300, bbox_inches='tight')
plt.show()
