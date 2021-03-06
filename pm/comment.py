"""
Module with classes for comment-data for each blog.
Is called from within comment_thread.py
"""

import datetime
import logging
import re

import access_classes as ac
from text_functions import tokenize

SETTINGS, _, LOCATION = ac.load_settings()
CONVERT, *_ = ac.load_yaml("settings/author_convert.yaml")


class Comment(object):
    """
    Class for comment-data. Takes a dedicated parser-class as argument.
    """

    def __init__(self, parser_class, comment, thread_url):
        self.parser = parser_class(comment)
        self.parser.set_com_id()
        self.parser.set_com_type_and_depth()
        try:
            self.parser.set_com_author(self.parser.parse_fun)
            self.parser.set_author_url()
        except AttributeError:
            self.parser.set_author_and_author_url()
        self.parser.set_comment_and_time()
        # get sequence-number of comment (if available)
        if SETTINGS['find implicit references']:
            self.parser.get_seq_nr(
                self.parser.node_attr['com_content'],
                thread_url)
        # make list of child-comments (only id's)
        self.parser.set_child_ids()
        # adding thread_url
        self.parser.node_attr['com_thread'] = thread_url
        self.parser.join_content()
        self.parser.tokenize_content()

    @property
    def data(self):
        return self.parser.com_id, self.parser.node_attr


class Parser(object):
    "Generic parser for comment"

    def __init__(self, comment):
        self.comment = comment
        self.node_attr = {}
        self.com_id = None

    @staticmethod
    def get_conv_author(comment, parse_fun):
        "Parses comment to find author, and converts to avoid duplicates"
        try:
            com_author = parse_fun(comment)
        except AttributeError as err:
            logging.warning("%s, %s", err, comment)
            com_author = "Unable to resolve"
        if com_author in CONVERT:
            com_author = CONVERT[com_author]
        else:  # no redundant spaces when converted
            com_author = re.sub(' +', ' ', com_author.strip())
        return com_author

    @staticmethod
    def get_seq_nr(content, url):
        "Looks for numbers in comments (implicit refs)"
        if url.path.split("/")[-2] not in SETTINGS["implicit_refs"]:
            seq_nr = None
        else:
            pattern = re.compile(r"\(\d+\)|\d+.\d*")
            content = "\n".join(content)
            matches = pattern.findall(content)
            try:
                seq_nr = matches[0]
                if seq_nr.startswith('('):
                    seq_nr = [int(seq_nr.strip("(|)"))]
                elif "." in seq_nr:
                    seq_nr = [int(i) for i in seq_nr.split(".") if i]
                else:
                    seq_nr = None
            except IndexError:
                seq_nr = None
        return seq_nr

    @staticmethod
    def parse_timestamp(time_stamp, date_format):
        "Parses time_stamp to datetime object."
        try:
            time_stamp = datetime.datetime.strptime(
                time_stamp, date_format)
        except ValueError as err:
            logging.warning("%s: datetime failed", err)
            print(time_stamp)
        return time_stamp

    def join_content(self):
        "Joins the lines in com_content."
        self.node_attr['com_content'] = " ".join(self.node_attr['com_content'])

    def tokenize_content(self):
        "Tokenizes com_content"
        self.node_attr['com_tokens'] = tokenize(self.node_attr['com_content'])


class StandardParser(Parser):
    "Class witharsing-methods is common to all blogs, except SbSeminar"

    def set_com_type_and_depth(self):
        "Sets comment-type and depth to node_attr"
        com_class = self.comment.get("class")
        self.node_attr["com_type"] = com_class[0]
        self.node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))

    def set_com_author(self, parsefun):
        "sets com_author to node_attr."
        self.node_attr['com_author'] = self.get_conv_author(
            self.comment, parsefun)


class PolymathCommentParser(StandardParser):
    "Parser for comment on Polymath-blog"

    def set_com_id(self):
        "sets com_id"
        self.com_id = self.comment.get("id")

    @staticmethod
    def parse_fun(comment):
        "Parsing function needed to find author."
        return comment.find("cite").find("span").text

    def set_comment_and_time(self):
        "Sets content and time of content to node_attr"
        com_all_content = [item.text for item in self.comment.find(
            "div", {"class": "comment-author vcard"}).find_all("p")]
        time_stamp = com_all_content.pop().split("—")[1].split(
            "\n")[0].strip()
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y @ %I:%M %p")
        self.node_attr['com_content'] = com_all_content

    def set_author_url(self):
        "Sets author_url to node_attr if it exists"
        try:
            com_author_url = self.comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          self.node_attr['com_author'])
            com_author_url = None
        self.node_attr['com_author_url'] = com_author_url

    def set_child_ids(self):
        "Adds id's of child-comments to node_attr"
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            child_comments = self.comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            self.node_attr['com_children'] = [child.get("id") for child in
                                              child_comments]
        except AttributeError:
            self.node_attr['com_children'] = []


class GilkalaiCommentParser(StandardParser):
    "Comment-data for comments on Kalai blog"

    def set_com_id(self):
        "sets com_id"
        self.com_id = self.comment.find("div").get("id")

    @staticmethod
    def parse_fun(comment):
        "Parsing function needed to find author."
        return comment.find("cite").text.strip()

    def set_comment_and_time(self):
        "Sets content and time of content to node_attr"
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find(
                "div", {"class": "comment-body"}).find_all("p")]
        time_stamp = self.comment.find(
            "div", {"class": "comment-meta commentmetadata"}).text.strip()
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y at %I:%M %p")

    def set_author_url(self):
        "Sets author_url to node_attr if it exists"
        try:
            self.node_attr['com_author_url'] = self.comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          self.node_attr['com_author'])
            self.node_attr['com_author_url'] = None

    def set_child_ids(self):
        "Adds id's of child-comments to node_attr"
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            child_comments = self.comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            self.node_attr['com_children'] = [
                child.find("div").get("id") for child in child_comments]
        except AttributeError:
            self.node_attr['com_children'] = []


class GowersCommentParser(StandardParser):
    "Comment-data for comments on Gowers-blog"

    def set_com_id(self):
        "sets com_id"
        self.com_id = self.comment.get("id")

    @staticmethod
    def parse_fun(comment):
        "Parsing function needed to find author."
        return comment.find("cite").text.strip()

    def set_comment_and_time(self):
        "Sets content and time of content to node_attr"
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find_all("p")]
        time_stamp = self.comment.find("small").find("a").text
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y at %I:%M %p")

    def set_author_url(self):
        "Sets author_url to node_attr if it exists"
        try:
            self.node_attr['com_author_url'] = self.comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          self.node_attr['com_author'])
            self.node_attr['com_author_url'] = None

    def set_child_ids(self):
        "Adds id's of child-comments to node_attr"
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            child_comments = self.comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            self.node_attr['com_children'] = [
                child.get("id") for child in child_comments]
        except AttributeError:
            self.node_attr['com_children'] = []


class MixonCommentParser(StandardParser):
    "Comment-data for comments on Mixon's blog"

    def set_com_id(self):
        "sets com_id"
        self.com_id = self.comment.get("id")

    @staticmethod
    def parse_fun(comment):
        "parsing function needed to find author"
        return comment.find(
            "div", {"class": "comment-author vcard"}).find("b").text

    def set_comment_and_time(self):
        "Sets content and time of content to node_attr"
        this_comment = self.comment.find("div",
                                         {"class": "comment-content"})
        if this_comment:  # do not proceed if None
            self.node_attr['com_content'] = [item.text for item in
                                             this_comment.find_all("p")]
            this_comment_data = self.comment.find(
                "div", {"class": "comment-metadata"})
            time_stamp = this_comment_data.time.text.strip()
            self.node_attr['com_timestamp'] = self.parse_timestamp(
                time_stamp, "%B %d, %Y at %I:%M %p")
        else:
            self.node_attr['com_content'] = []
            self.node_attr['com_timestamp'] = None

    def set_author_url(self):
        "Sets author_url to node_attr if it exists"
        try:
            self.node_attr['com_author_url'] = self.comment.find(
                "div", {"class": "comment-author vcard"}).find(
                    "a", {"class": "url"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          self.node_attr['com_author'])
            self.node_attr['com_author_url'] = None

    def set_child_ids(self):
        "Adds id's of child-comments to node_attr"
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            child_comments = self.comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            self.node_attr['com_children'] = [
                child.get("id") for child in child_comments]
        except AttributeError:
            self.node_attr['com_children'] = []


class SBSCommentParser(Parser):
    "Comment-data for comments on SBS-blog"

    def set_com_id(self):
        "sets com_id"
        self.com_id = self.comment.get("id")

    def set_com_type_and_depth(self):
        "Sets comment-type and depth to node_attr"
        com_class = self.comment.get("class")
        self.node_attr['com_type'] = com_class[0]
        self.node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))

    def set_comment_and_time(self):
        "Sets content and time of content to node_attr"
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find(
                "div", {"class": "comment-content"}).find_all("p")]
        time_stamp = self.comment.find(
            "div", {"class": "comment-metadata"}).find("time").get(
                "datetime")
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%Y-%m-%dT%H:%M:%S+00:00")

    @staticmethod
    def parse_fun(comment):
        "Parsing function needed to find author."
        return comment.find("cite").text.strip()

    def set_author_and_author_url(self):
        "Sets author and author_url to node_attr"
        com_author_and_url = self.comment.find(
            "div", {"class": "comment-author"}).find(
                "cite", {"class": "fn"})
        try:
            com_author = com_author_and_url.find("a").text
            self.node_attr['com_author_url'] = com_author_and_url.find(
                "a").get("href")
        except AttributeError:
            try:
                com_author = com_author_and_url.text
            except AttributeError:
                logging.debug("Could not resolve author_url for %s",
                              com_author)
                com_author = "unable to resolve"
            finally:
                self.node_attr['com_author_url'] = None
        self.node_attr['com_author'] = CONVERT[com_author] if\
            com_author in CONVERT else com_author

    def set_child_ids(self):
        "Adds id's of child-comments to node_attr (void here)"
        self.node_attr['com_children'] = []


class TaoCommentParser(StandardParser):
    "Parser for comment on Tao-blog"

    def set_com_id(self):
        "sets com_id"
        self.com_id = self.comment.get("id")

    @staticmethod
    def parse_fun(comment):
        "Parsing function needed to find author."
        return comment.find("p", {"class": "comment-author"}).text

    def set_comment_and_time(self):
        "Sets content and time of content to node_attr"
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find_all("p")][2:]
        time_stamp = self.comment.find(
            "p", {"class": "comment-permalink"}).find("a").text
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%d %B, %Y at %I:%M %p")

    def set_author_url(self):
        "Sets author_url to node_attr if it exists"
        try:
            self.node_attr['com_author_url'] = self.comment.find(
                "p", {"class": "comment-author"}).find("a").get("href")
        except AttributeError:
            logging.debug(
                "Could not resolve author_url for %s",
                self.node_attr['com_author'])
            self.node_attr['com_author_url'] = None

    def set_child_ids(self):
        "Adds id's of child-comments to node_attr"
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            if self.comment.next_sibling.next_sibling['class'] == ['children']:
                child_comments = self.comment.next_sibling.next_sibling.\
                    find_all("div", {"class": "comment"})
                self.node_attr['com_children'] = [
                    child.get("id") for child in child_comments if
                    depth_search in child["class"]]
            else:
                self.node_attr['com_children'] = []
        except (AttributeError, TypeError):
            self.node_attr['com_children'] = []
