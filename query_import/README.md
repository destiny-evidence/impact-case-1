python query_import/import_sample.py --config=.conf/secret.env --ids=data/random_ids.txt --target=data/20260429-openalex.jsonl --ensure-abstract --project-id=db6ee519-afb5-4813-822b-bfbc7dfd2237

cd /data/workspace/nacsos-support
uv run nacsos import ACADEMIC --source 20260429-openalex.jsonl --project-id='db6ee519-afb5-4813-822b-bfbc7dfd2237' --config-file /data/nacsos2/nacsos-core/config/server.env --import-id='b250dc09-e15f-43f7-9d56-8b80ed747d1a'
