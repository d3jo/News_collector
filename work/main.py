#!/usr/bin/env python3
"""
Privacy News Monitor - 개인정보 보호 동향 모니터링 서비스

지난 48시간 동안 해외에서 발생한 개인정보 관련 주요 뉴스를 수집하고
요약하여 제공하는 서비스입니다.

사용법:
    python main.py [--output OUTPUT_FILE] [--format {detailed|category|simple}] [--use-emoji]

환경 변수:
    GNEWS_API_KEY: GNews API 키
    NEWSAPI_KEY: NewsAPI 키
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from privacy_news_algorithm.collector import create_collector
from privacy_news_algorithm.formatter import create_formatter


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="개인정보 보호 동향 모니터링 서비스",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py                           # 기본 실행 (터미널 출력)
    python main.py --output report.md        # 파일로 저장
    python main.py --format category         # 카테고리별 그룹화
    python main.py --format simple           # 간단한 목록 형태
        """
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="출력 파일 경로 (지정하지 않으면 터미널에 출력)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["detailed", "category", "simple"],
        default="detailed",
        help="출력 형식 (detailed: 상세, category: 카테고리별, simple: 간단)"
    )
    parser.add_argument(
        "--use-emoji",
        action="store_true",
        help="카테고리 라벨에 이모지 사용"
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="API 키 없이 샘플 데이터 사용"
    )

    args = parser.parse_args()

    # API 키 확인
    gnews_key = os.environ.get("GNEWS_API_KEY")
    newsapi_key = os.environ.get("NEWSAPI_KEY")

    # 수집기 생성
    collector = create_collector(gnews_api_key=gnews_key, newsapi_key=newsapi_key)

    # 뉴스 수집
    print("개인정보 보호 관련 뉴스를 수집 중입니다...", file=sys.stderr)


    # 포맷터 생성 및 출력 생성
    formatter = create_formatter(use_emoji=args.use_emoji)

    return 0


if __name__ == "__main__":
    sys.exit(main())
