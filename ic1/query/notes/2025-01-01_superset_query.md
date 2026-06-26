
* We still have original set of works (latest update from Oct 2024)
  * Of those, ~3k are manually annotated as relevant
  * ~40k were predicted as relevant (trained on human annotations)
* We translated the query to dimensions and got all those records
* Using abstracts from these two sources for gap-filling did not 
  result in too many updates (forgot to track, was around 30-50k)

https://liveuclac.sharepoint.com/:p:/r/sites/DESTINY/_layouts/15/Doc.aspx?sourcedoc=%7BE5256BE1-81BC-4287-B86D-48083CCAFC3D%7D&file=OpenAlex.pptx&wdOrigin=TEAMS-WEB.teams_ns.rwc&action=edit&mobileredirect=true

# Summary numbers
On NACSOS: 1,290,164 (in Oct 2024)
Trivial query adaptation
  * 1,065,881 query only
  *   559,515 not oa
  *    90,043 springer|elsevier
  *    11,371 not oa & springer|elsevier

after replacing wildcards and simplifying
  * 1,334,707 query only
  *   709,750 not oa
  *   112,715 springer|elsevier
  *    16,506 not oa & springer|elsevier

Estimated query time:  
1,334,707 records, 200 per page -> ~6674 pages  
max 10 requests per second -> ~667 seconds -> ~11min

# Overlaps
```
OpenAlex has 1,219,472 records
  NACSOS has 1,239,989 records
   > Union: 1,690,854 | intersect: 768,607 | OpenAlex exclusive: 450,865 | NACSOS exclusive 471,382
  NACSOS (rel) has 2,146 records
   > Union: 1,219,976 | intersect: 1,642 | OpenAlex exclusive: 1,217,830 | NACSOS (rel) exclusive 504
  Predicted has 42,881 records
   > Union: 1,229,810 | intersect: 32,543 | OpenAlex exclusive: 1,186,929 | Predicted exclusive 10,338
NACSOS has 1,239,989 records
  OpenAlex has 1,219,472 records
   > Union: 1,690,854 | intersect: 768,607 | NACSOS exclusive: 471,382 | OpenAlex exclusive 450,865
  NACSOS (rel) has 2,146 records
   > Union: 1,239,990 | intersect: 2,145 | NACSOS exclusive: 1,237,844 | NACSOS (rel) exclusive 1
  Predicted has 42,881 records
   > Union: 1,241,254 | intersect: 41,616 | NACSOS exclusive: 1,198,373 | Predicted exclusive 1,265
NACSOS (rel) has 2,146 records
  OpenAlex has 1,219,472 records
   > Union: 1,219,976 | intersect: 1,642 | NACSOS (rel) exclusive: 504 | OpenAlex exclusive 1,217,830
  NACSOS has 1,239,989 records
   > Union: 1,239,990 | intersect: 2,145 | NACSOS (rel) exclusive: 1 | NACSOS exclusive 1,237,844
  Predicted has 42,881 records
   > Union: 43,543 | intersect: 1,484 | NACSOS (rel) exclusive: 662 | Predicted exclusive 41,397
Predicted has 42,881 records
  OpenAlex has 1,219,472 records
   > Union: 1,229,810 | intersect: 32,543 | Predicted exclusive: 10,338 | OpenAlex exclusive 1,186,929
  NACSOS has 1,239,989 records
   > Union: 1,241,254 | intersect: 41,616 | Predicted exclusive: 1,265 | NACSOS exclusive 1,198,373
  NACSOS (rel) has 2,146 records
   > Union: 43,543 | intersect: 1,484 | Predicted exclusive: 41,397 | NACSOS (rel) exclusive 662
```