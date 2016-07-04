"""
Module for mixin classes for accessor methods
common to comment_thread and multi_comment_thread
"""

import re
import yaml
import matplotlib.pyplot as plt
import networkx as nx
import nltk


def show_or_save(show):
    """Shows or saves plot"""
    if show:
        plt.show()
    else:
        filename = input("Give filename: ")
        filename += ".png"
        plt.savefig(filename)


class ThreadAccessMixin(object):
    """description"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_name = {}

    def comment_report(self, com_id):
        """Takes node-id, and returns dict with report about node."""
        the_node = self.graph.node[com_id]  # dict
        the_author = the_node["com_author"]  # string
        descendants = nx.descendants(self.graph, com_id)
        pure_descendants = [i for i in descendants if
                            self.graph.node[i]['com_author'] != the_author]
        direct_descendants = self.graph.out_degree(com_id)
        return {
            "author": the_author,
            "level of comment": the_node["com_depth"],
            "replies (direct)": direct_descendants,
            "replies (all)": len(descendants),
            "replies (own excl.)": len(pure_descendants)
        }

    def print_nodes(self, *select):
        """takes nodes-id(s), and prints out node-data as yaml. No output."""
        if select:
            select = list(self.node_name.keys()) if select[0].lower() == "all"\
                else select
            for comment in select:  # do something if comment does not exist!
                print("com_id:", comment)
                try:
                    print(yaml.dump(self.graph.node[comment],
                                    default_flow_style=False))
                except KeyError as err:
                    print(err, "not found")
                print("------------------------------------------------------")
        else:
            print("No nodes were selected")

    @staticmethod
    def tokenize_and_stem(text, notext_pattern=re.compile("[^a-zA-Z0-9]")):
        """
        takes unicode-text and returns tokenized and stemmed
        and just tokenized content
        """
        filtered_text = notext_pattern.sub(" ", text)
        tokens = [word.lower() for sent in nltk.sent_tokenize(filtered_text)
                  for word in nltk.word_tokenize(sent)]
        stemmer = nltk.stem.snowball.SnowballStemmer("english")
        stems = [stemmer.stem(token) for token in tokens]
        return tokens, stems

    @classmethod
    def strip_proppers_POS(cls, text):
        tagged = nltk.tag.pos_tag(text.split())  # use NLTK's speech tagger
        non_propernouns = [word for word, pos in tagged if
                           pos != 'NNP' and pos != 'NNPS']
        return non_propernouns

