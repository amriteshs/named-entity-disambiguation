# Named-Entity-Disambiguation
Aims at assigning unique identities (i.e., entities, such as Persons, Locations and Organizations etc.) to the mention (i.e., a substring/span of the sentence that refer to an entity) identified in the text.

## Part-1
1. TF-IDF Index Construction for Entities and Tokens
2. Split the Query into Entities and Tokens
3. Query Score Computation

## Part-2
Given a document, a mention span within the document, and a collection of candidate entities for each mention alongwith corresponding entity description pages, generated a learning-to-rank model to rank the candidate entities corresponding to each mention in such a way that the Ground Truth Entity is ranked higher than the false candidates. The XGBoost classifier has been used to build the learning-to-rank model.
