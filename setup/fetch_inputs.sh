#!/usr/bin/env bash
# Fetch only the pinned TweetEval sentiment split and BoAmps schema. No model weights.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
tweet_rev=4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66
schema_rev=af1680c56ce2cefd5777fcac1c57591b685d6be5
mkdir -p data/tweeteval "schema/$schema_rev"
fetch() {
    local url=$1 dst=$2 exp tmp
    exp=$(awk -v p="$dst" '$2 == p { print $1 }' setup/inputs.sha256)
    [[ ${#exp} == 64 ]]
    if [[ -f "$dst" ]] && printf '%s  %s\n' "$exp" "$dst" | sha256sum -c --status -; then
        return
    fi
    tmp=$(mktemp "${dst}.tmp.XXXXXX")
    if curl --fail --silent --show-error --location --retry 2 --max-time 120 "$url" -o "$tmp" \
        && printf '%s  %s\n' "$exp" "$tmp" | sha256sum -c -; then
        mv -- "$tmp" "$dst"
    else
        rm -f -- "$tmp"
        return 1
    fi
}
for f in mapping.txt {train,val,test}_{text,labels}.txt; do
    fetch "https://raw.githubusercontent.com/cardiffnlp/tweeteval/$tweet_rev/datasets/sentiment/$f" "data/tweeteval/$f"
done
for f in {report,algorithm,dataset,measure,hardware}_schema.json; do
    fetch "https://raw.githubusercontent.com/Boavizta/BoAmps/$schema_rev/model/$f" "schema/$schema_rev/$f"
done
sha256sum -c setup/inputs.sha256
