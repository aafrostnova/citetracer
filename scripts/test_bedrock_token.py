"""Quick check: does the Bedrock bearer token in config.json work?

Tests the token used by the generate_mutations.py path:
  1. Load config.json
  2. Try verification_llm.bedrock.bearer_token, fall back to entry_extraction.bedrock.bearer_token
  3. Call Bedrock converse with the configured model_id
  4. Print SUCCESS / FAILED with reason

Usage:
    python scripts/test_bedrock_token.py
    python scripts/test_bedrock_token.py --token ABSK... --model qwen.qwen3-vl-235b-a22b --region us-east-1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", help="Bearer token to test (overrides config.json)")
    parser.add_argument("--model", help="Model id (overrides config.json)")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--prompt", default="Say 'hello' and nothing else.")
    args = parser.parse_args()

    # Resolve token/model
    token = args.token
    model_id = args.model
    region = args.region

    if not token or not model_id:
        from apps.pdf_checker.config import load_pdf_checker_config
        cfg = load_pdf_checker_config()
        ver = cfg.verification_llm.bedrock
        ext = cfg.entry_extraction.bedrock
        token = token or ver.bearer_token or ext.bearer_token
        model_id = model_id or str(ver.model_id or ext.model_id or "")
        region = args.region or ver.region or ext.region or "us-east-1"

    if not token:
        print("FAIL: no bearer_token found in config.json or --token flag")
        return 2
    if not model_id:
        print("FAIL: no model_id found in config.json or --model flag")
        return 2

    print(f"Region:   {region}")
    print(f"Model:    {model_id}")
    print(f"Token:    {token[:20]}... ({len(token)} chars)")
    print(f"Prompt:   {args.prompt!r}")
    print("---")

    import os
    import boto3
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    client = boto3.client("bedrock-runtime", region_name=region)

    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": args.prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 64},
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    reply = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    print("SUCCESS")
    print(f"  reply:  {reply.strip()[:200]!r}")
    print(f"  usage:  input={usage.get('inputTokens')} output={usage.get('outputTokens')} total={usage.get('totalTokens')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
