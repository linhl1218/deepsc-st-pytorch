# -*- coding: utf-8 -*-
"""
Created on 2023/09/27 14:17
@author: Hailong Lin
"""
import logging

def create_logger(filename, ltype="f", name="logs"):
    """
    Create a logger
    Args:
        filename (str): logs save file path
        ltype (str, optional): _description_. Defaults to "f".
        "f" logger only save to file, "s" only print, "a" file and print
        name (str, optional): _description_. Defaults to "logs".

    Returns:
        _type_: _description_
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if ltype in ("f", "a"):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if ltype in ("s", "a"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

if __name__=="__main__":
    test_logger = create_logger("./logs/test.log", ltype="a")
    test_logger.info("This is a INFO message")
