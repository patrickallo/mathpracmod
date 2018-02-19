"""
Module with classes for comment-data for each blog.
Is called from within comment_thread.py
"""

import datetime
import logging
import re

import access_classes as ac

SETTINGS, _, LOCATION = ac.load_settings()
CONVERT, *_ = ac.load_yaml("settings/author_convert.yaml")


class Comment(object):
    """
    Parent-class for all comment objects
    """

    def __init__(self, comment):
        self.comment = comment
        self.node_attr = {}
        self.com_id = None

    def __call__(self):
        print("Comment called.")
        return self.com_id, self.node_attr

    @staticmethod
    def get_conv_author(comment, parse_fun):
        """Parses comment to find author, and converts to avoid duplicates"""
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
        """Looks for numbers in comments (implicit refs)"""
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
        """Parses time_stamp to datetime object."""
        try:
            time_stamp = datetime.datetime.strptime(
                time_stamp, date_format)
        except ValueError as err:
            logging.warning("%s: datetime failed", err)
            print(time_stamp)
        return time_stamp


class StandardComment(Comment):
    "Class which groups what is common to all blogs, except SbSeminar"

    def __init__(self, comment):
        super(StandardComment, self).__init__(comment)

    def __get_com_id(self):
        raise NotImplementedError

    def set_com_type_and_depth(self):
        com_class = self.comment.get("class")
        self.node_attr["com_type"] = com_class[0]
        self.node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))

    def set_com_author(self, parsefun):
        self.node_attr['com_author'] = self.get_conv_author(
            self.comment, parsefun)


class CommentPolymath(StandardComment):
    "Comment-data for comments on Polymath-blog"

    def __init__(self, comment, thread_url):
        super(CommentPolymath, self).__init__(comment)
        self.com_id = self.__get_com_id()
        self.set_com_type_and_depth()
        self.set_com_author(self.__parse_fun)
        self.__process_comment_and_time()
        self.__set_author_url()
        # get sequence-number of comment (if available)
        if SETTINGS['find implicit references']:
            self.node_attr['seq_nr'] = self.get_seq_nr(
                self.node_attr['com_content'],
                thread_url)
        # make list of child-comments (only id's)
        self.__set_child_ids()
        # adding thread_url
        self.node_attr['com_thread'] = thread_url

    def __get_com_id(self):
        return self.comment.get("id")

    @staticmethod
    def __parse_fun(comment):
        return comment.find("cite").find("span").text

    def __process_comment_and_time(self):
        com_all_content = [item.text for item in self.comment.find(
            "div", {"class": "comment-author vcard"}).find_all("p")]
        time_stamp = com_all_content.pop().split("—")[1].split(
            "\n")[0].strip()
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y @ %I:%M %p")
        self.node_attr['com_content'] = com_all_content

    def __set_author_url(self):
        try:
            com_author_url = self.comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          self.node_attr['com_author'])
            com_author_url = None
        self.node_attr['com_author_url'] = com_author_url

    def __set_child_ids(self):
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            child_comments = self.comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            self.node_attr['com_children'] = [child.get("id") for child in
                                              child_comments]
        except AttributeError:
            self.node_attr['com_children'] = []


class CommentGilkalai(StandardComment):
    "Comment-data for comments on Kalai blog"

    def __init__(self, comment, thread_url):
        super(CommentGilkalai, self).__init__(comment)
        self.com_id = self.__get_com_id()
        self.set_com_type_and_depth()
        self.set_com_author(self.__parse_fun)
        self.__process_comment_and_time()
        self.__set_author_url()
        # get sequence-number of comment (if available)
        if SETTINGS['find implicit references']:
            self.node_attr['seq_nr'] = self.get_seq_nr(
                self.node_attr['com_content'],
                thread_url)
        # make list of child-comments (only id's)
        self.__set_child_ids()
        # adding thread_url
        self.node_attr['com_thread'] = thread_url

    def __get_com_id(self):
        return self.comment.find("div").get("id")

    @staticmethod
    def __parse_fun(comment):
        return comment.find("cite").text.strip()

    def __process_comment_and_time(self):
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find(
                "div", {"class": "comment-body"}).find_all("p")]
        time_stamp = self.comment.find(
            "div", {"class": "comment-meta commentmetadata"}).text.strip()
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y at %I:%M %p")

    def __set_author_url(self):
        try:
            self.node_attr['com_author_url'] = self.comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          self.node_attr['com_author'])
            self.node_attr['com_author_url'] = None

    def __set_child_ids(self):
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            child_comments = self.comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            self.node_attr['com_children'] = [
                child.find("div").get("id") for child in child_comments]
        except AttributeError:
            self.node_attr['com_children'] = []


class CommentGowers(StandardComment):
    "Comment-data for comments on Gowers-blog"

    def __init__(self, comment, thread_url):
        super(CommentGowers, self).__init__(comment)
        self.com_id = self.__get_com_id()
        self.set_com_type_and_depth()
        self.set_com_author(self.__parse_fun)
        self.__process_comment_and_time()
        self.__set_author_url()
        # get sequence-number of comment (if available)
        if SETTINGS['find implicit references']:
            self.node_attr['seq_nr'] = self.get_seq_nr(
                self.node_attr['com_content'],
                thread_url)
        # make list of child-comments (only id's)
        self.__set_child_ids()
        # adding thread_url
        self.node_attr['com_thread'] = thread_url

    def __get_com_id(self):
        return self.comment.get("id")

    @staticmethod
    def __parse_fun(comment):
        return comment.find("cite").text.strip()

    def __process_comment_and_time(self):
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find_all("p")]
        time_stamp = self.comment.find("small").find("a").text
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y at %I:%M %p")

    def __set_author_url(self):
        try:
            self.node_attr['com_author_url'] = self.comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          self.node_attr['com_author'])
            self.node_attr['com_author_url'] = None

    def __set_child_ids(self):
        try:
            depth_search = "depth-" + str(self.node_attr['com_depth'] + 1)
            child_comments = self.comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            self.node_attr['com_children'] = [
                child.get("id") for child in child_comments]
        except AttributeError:
            self.node_attr['com_children'] = []


class CommentSBS(Comment):
    "Comment-data for comments on Gowers-blog"

    def __init__(self, comment, thread_url):
        super(CommentSBS, self).__init__(comment)
        self.com_id = self.__get_com_id()
        self.__set_com_type_and_depth()
        self.__process_comment_and_time()
        self.__set_author_and_author_url()
        # get sequence-number of comment (if available)
        if SETTINGS['find implicit references']:
            self.node_attr['seq_nr'] = self.get_seq_nr(
                self.node_attr['com_content'],
                thread_url)
        # make list of child-comments (only id's)
        self.node_attr['com_children'] = []
        # adding thread_url
        self.node_attr['com_thread'] = thread_url

    def __get_com_id(self):
        return self.comment.get("id")

    def __set_com_type_and_depth(self):
        com_class = self.comment.get("class")
        self.node_attr['com_type'] = com_class[0]
        self.node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))

    def __process_comment_and_time(self):
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find(
                "div", {"class": "comment-content"}).find_all("p")]
        time_stamp = self.comment.find(
            "div", {"class": "comment-metadata"}).find("time").get(
                "datetime")
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%Y-%m-%dT%H:%M:%S+00:00")

    def __set_author_and_author_url(self):
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


class CommentTao(StandardComment):
    "Comment-data for comments on Tao-blog"

    def __init__(self, comment, thread_url):
        super(CommentTao, self).__init__(comment)
        self.com_id = self.__get_com_id()
        self.set_com_type_and_depth()
        self.set_com_author(self.__parse_fun)
        self.__process_comment_and_time()
        self.__set_author_url()
        # get sequence-number of comment (if available)
        if SETTINGS['find implicit references']:
            self.node_attr['seq_nr'] = self.get_seq_nr(
                self.node_attr['com_content'],
                thread_url)
        # make list of child-comments (only id's)
        self.__set_child_ids()
        # adding thread_url
        self.node_attr['com_thread'] = thread_url

    def __get_com_id(self):
        return self.comment.get("id")

    @staticmethod
    def __parse_fun(comment):
        return comment.find("p", {"class": "comment-author"}).text

    def __process_comment_and_time(self):
        self.node_attr['com_content'] = [
            item.text for item in self.comment.find_all("p")][2:]
        time_stamp = self.comment.find(
            "p", {"class": "comment-permalink"}).find("a").text
        self.node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%d %B, %Y at %I:%M %p")

    def __set_author_url(self):
        try:
            self.node_attr['com_author_url'] = self.comment.find(
                "p", {"class": "comment-author"}).find("a").get("href")
        except AttributeError:
            logging.debug(
                "Could not resolve author_url for %s",
                self.node_attr['com_author'])
            self.node_attr['com_author_url'] = None

    def __set_child_ids(self):
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
