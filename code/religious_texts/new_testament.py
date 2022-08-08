import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_bible():
    """
    Routine to get the new testament saved into format
    """
    bible_source = "https://www.gutenberg.org/files/10/10-h/10-h.htm"
    r = requests.get(bible_source)
    soup = BeautifulSoup(r.content, "html5lib")

    verses = [i.contents[0] for i in soup.find_all("p")]
    chapter_verse = [i.split(" ")[0] for i in verses]

    for i, cv in enumerate(chapter_verse):
        if len(cv.split(":")) < 2:
            verses.pop(i)
            chapter_verse.pop(i)
    for i, cv in enumerate(chapter_verse):
        if len(cv.split(":")) < 2:
            verses.pop(i)
            chapter_verse.pop(i)

    chapter_num = [i.split(":")[0] for i in chapter_verse]
    verse_num = [i.split(":")[1] for i in chapter_verse]

    verses = [" ".join(i.split(" ")[1:]) for i in verses]

    bible = pd.DataFrame(
        zip(chapter_num, verse_num, verses),
        columns=["chapter_num", "verse_num", "verse"],
    )
    bible.to_csv(
        "../data/Christianity/Bible/New_Testament/newtestament.csv", index=False
    )
    return bible
