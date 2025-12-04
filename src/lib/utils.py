import json
from typing import List
import zipfile
from pdfminer.high_level import extract_text
import os
from dotenv import load_dotenv
from pathlib import Path
import io
import re
import csv
import sys
from diskcache import Cache
import time
import requests
from bs4 import BeautifulSoup
import random
import string
from urllib.parse import urlparse, parse_qs

dc = Cache("work/cache")

GENERIC_STOP_WORDS = {
    "experience",
    "candidate",
    "ability",
    "knowledge",
    "years",
    "work",
    "team",
    "role",
    "requirements",
    "responsibilities",
    "degree",
    "skills",
    "environment",
    "projects",
    "company",
    "opportunity",
}
currenttimemillis = lambda: int(round(time.time() * 1000))


def extract_json_from_crew_output(crew_output: str) -> dict:
    # find location of : ```json
    start = crew_output.find("```json") + len("```json")
    end = crew_output.find("```", start)
    return json.loads(crew_output[start:end])


def download_file(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        return str(e)


def get_text_from_url(url):
    job_description_class = "show-more-less-html__markup"
    bs_obj = BeautifulSoup(download_file(url), "html.parser")
    job_description = bs_obj.find(class_="show-more-less-html__markup").get_text()
    return job_description


def extract_text_from_file(file_path):
    try:
        print(f"Extracting text from {file_path}")
        # Extracts all text from the file
        with open(file_path, "r") as f:
            text = f.read()
        clean_text = " ".join(text.split())
        return clean_text
    except Exception as e:
        return str(e)


def extract_text_from_pdf(pdf_path):
    try:
        text_file_path = Path("parsed_files") / f"{Path(pdf_path).name}.txt"
        print(f"Extracting text from {pdf_path}")
        # Extracts all text from the PDF file
        text = extract_text(pdf_path)
        # there appear to be lots of tabs and junk in the pdf extraction
        # this somewhat cleans that up
        clean_text = " ".join(text.split())
        with open(text_file_path, "w") as f:
            f.write(clean_text)
            print(f"Wrote parsed content to {text_file_path}")
        return clean_text
    except Exception as e:
        return str(e)


# initialize matcher with a vocab


# def extract_name(resume_text):
#     """Extracts the name of a person from resume text using pattern matching."""
#     # Pre-process the text to remove extra whitespace and newlines
#     cleaned_text = " ".join(resume_text[:200].split())
#     doc = nlp(cleaned_text)
#     matcher = Matcher(nlp.vocab)

#     # Pattern to find two consecutive proper nouns (FirstName LastName)
#     pattern = [{"POS": "PROPN"}, {"POS": "PROPN"}]
#     matcher.add("NAME", [pattern])
#     matches = matcher(doc)

#     # Return the first match found
#     if matches:
#         return doc[matches[0][1] : matches[0][2]].text.strip()
#     return None


# def parse_resume_get_name_email_phone(text):
#     doc = nlp(text[:200])
#     data = {"name": "", "email": "", "phone_number": ""}

#     # 1 get name
#     name = extract_name(text)
#     if name:
#         data["name"] = name
#         print(f"Name: {data['name']}")
#     # 2. Extract Email using Regex (Simpler than NLP)
#     email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
#     email_match = re.search(email_pattern, text)
#     if email_match:
#         data["email"] = email_match.group(0)
#         print(f"Email: {data['email']}")

#     # 3. Extract Phone Number
#     phone_pattern = r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
#     phone_match = re.search(phone_pattern, text)
#     if phone_match:
#         data["phone_number"] = phone_match.group(0)
#         print(f"Phone Number: {data['phone_number']}")

#     return data


# def parse_resume_get_skills(text, target_skills: List[str]):
#     doc = nlp(text)
#     data = {
#         "skills": [],
#     }

#     # 3. Extract Skills (Keyword Matching)
#     # In a real app, load this list from a database or use a Skill Ontology
#     # target_skills = ["Python", "SQL", "Machine Learning", "Communication", "Java", "React"]

#     # Normalize text to lowercase for matching
#     text_lower = text.lower()
#     for skill in target_skills:
#         if skill.lower() in text_lower:
#             data["skills"].append(skill)

#     return data


# A small "Blocklist" to ignore generic corporate words that are nouns but not skills
# You can expand this list over time
# GENERIC_STOP_WORDS = {
#     "experience",
#     "candidate",
#     "ability",
#     "knowledge",
#     "years",
#     "work",
#     "team",
#     "role",
#     "requirements",
#     "responsibilities",
#     "degree",
#     "skills",
#     "environment",
#     "projects",
#     "company",
#     "opportunity",
# }


# def read_skills_from_kaggle_file(cache: bool = True) -> set:
#     """
#     Reads 'job_skills.csv' from 'archive.zip' and returns its content.
#     """
#     zip_file_path = "archive.zip"
#     csv_file_name = "job_skills.csv"
#     dc_key = f"{zip_file_path}:{csv_file_name}"
#     if cache:
#         all_skills = dc.get(dc_key, None)
#         if all_skills is not None:
#             print(f"Using cached skills for {dc_key}")
#             return set(all_skills)
#     all_skills = set()

#     try:
#         with zipfile.ZipFile(zip_file_path, "r") as zf:
#             with zf.open(csv_file_name, "r") as csv_file:
#                 # Wrap the binary file in a TextIOWrapper to decode line-by-line, improving memory efficiency.
#                 text_stream = io.TextIOWrapper(csv_file, encoding="utf-8")
#                 csv_reader = csv.reader(text_stream)
#                 next(csv_reader)  # Skip header row

#                 for row in csv_reader:
#                     if len(row) > 1:
#                         # Use update() for in-place modification, which is more performant than union().
#                         skills_to_add = (skill.strip() for skill in row[1].split(","))
#                         all_skills.update(skills_to_add)

#         with open("work/skills.txt", "w") as f:
#             f.write("\n".join(all_skills))
#             print("Wrote skills to work/skills.txt")
#     except Exception as e:
#         dc.set(dc_key, list(all_skills))
#         return all_skills


# def extract_skills_from_text(text: str, skill_set: set = None) -> set:
#     """
#     Efficiently extracts skills from text using regex matching.

#     Args:
#         text: The text to extract skills from (resume or job description)
#         skill_set: Set of skills to search for. If None, loads from Kaggle file.

#     Returns:
#         Set of matched skills (normalized to lowercase)

#     Performance optimizations:
#     - Uses regex with word boundaries for accurate matching
#     - Case-insensitive matching via lowercase normalization
#     - Avoids creating large spaCy pipelines
#     - Uses set operations for fast lookups
#     """
#     if skill_set is None:
#         skill_set = read_skills_from_kaggle_file()

#     # Normalize text to lowercase for case-insensitive matching
#     text_lower = text.lower()

#     # Extract matched skills
#     matched_skills = set()

#     # Create a mapping of lowercase skills to original case for better matching
#     # This allows us to match case-insensitively but return consistent results
#     skill_map = {skill.lower(): skill for skill in skill_set}

#     # Sort skills by length (longest first) to match multi-word skills before single words
#     # This prevents "Machine Learning" from being split into "Machine" and "Learning"
#     sorted_skills = sorted(skill_map.keys(), key=len, reverse=True)

#     # Use regex with word boundaries for accurate matching
#     # This prevents partial word matches (e.g., "C" matching "CSS")
#     processed_skills = 0
#     total_skills = len(sorted_skills)
#     for skill_lower in sorted_skills:
#         processed_skills += 1
#         if processed_skills % 1000 == 0:
#             print(
#                 f"Processed {processed_skills}/{total_skills} {processed_skills / total_skills * 100:.2f}% skills"
#             )
#         # Escape special regex characters in the skill name
#         escaped_skill = re.escape(skill_lower)
#         # Use word boundaries (\b) to match whole words/phrases only
#         pattern = r"\b" + escaped_skill + r"\b"

#         if re.search(pattern, text_lower):
#             matched_skills.add(skill_lower)

#     return matched_skills


# def extract_skills_from_text_fast(text: str, skill_set: set = None) -> set:
#     """
#     Fast skill extraction using pre-compiled regex patterns with word boundaries.

#     This version uses compiled regex patterns for better performance while
#     maintaining accuracy with word boundary matching.

#     Args:
#         text: The text to extract skills from
#         skill_set: Set of skills to search for. If None, loads from Kaggle file.

#     Returns:
#         Set of matched skills (normalized to lowercase)
#     """
#     if skill_set is None:
#         raise ValueError("skill_set must be provided")

#     # Normalize text to lowercase
#     text_lower = text.lower()

#     # Pre-compile regex patterns for all skills (with caching for repeated calls)
#     # This is faster than compiling on each search
#     matched_skills = set()
#     total_skills = len(skill_set)
#     processed_skills = 0
#     for skill in skill_set:
#         processed_skills += 1
#         skill_lower = skill.lower()
#         # Escape special regex characters and add word boundaries
#         escaped_skill = re.escape(skill_lower)
#         pattern = r"\b" + escaped_skill + r"\b"

#         # Use re.search which is faster than re.findall when we just need to know if it exists
#         if re.search(pattern, text_lower):
#             matched_skills.add(skill_lower)
#         if processed_skills % 1000 == 0:
#             print(
#                 f"Processed {processed_skills}/{total_skills} {processed_skills / total_skills * 100:.2f}% skills"
#             )

#     return matched_skills


def identify_job_source(url: str) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    if url.startswith("http"):
        print(f"Identifying job source from URL: {url}")
        if "linkedin" in url:
            # example: https://www.linkedin.com/jobs/view/4308118213/?eBP=NON_CHARGEABLE_CHANNEL&refId=G%2BFao3NOlOSc8QUHNxJkvg%3D%3D&trackingId=aHsjCzSeKWJeghtTUZFbcQ%3D%3D&trk=flagship3_search_srp_jobs&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_jobs%3BN3X1wCt5SMqWow9PKjAieA%3D%3D&lici=aHsjCzSeKWJeghtTUZFbcQ%3D%3D
            job_id = url.split("/")[5]
            if job_id == "":
                job_id = random_id
                print(f"Job ID not found in URL: {url}. Using random ID: {job_id}")
                return {
                    "job_source": "LinkedIn",
                    "job_id": job_id,
                    "job_url": "",
                }
            return {
                "job_source": "LinkedIn",
                "job_id": job_id,
                "job_url": f"https://www.linkedin.com/jobs/view/{job_id}",
            }
        elif "indeed" in url:
            # example: https://www.indeed.com/?vjk=ec1c9b9378ad1a8e&advn=4418968771450209
            # get the query parameter vjk using urlparse
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            job_id = query_params.get("vjk", random_id)
            if job_id == "":
                job_id = random_id
                print(f"Job ID not found in URL: {url}. Using random ID: {job_id}")
                return {"job_source": "Indeed", "job_id": job_id, "job_url": ""}

            return {
                "job_source": "Indeed",
                "job_id": job_id,
                "job_url": "https://www.indeed.com/?vjk={job_id}",
            }
        else:
            # Generate a random 10-character alphanumeric string
            return {"job_source": "Unknown", "job_id": random_id}
    else:
        return {"job_source": "Unknown", "job_id": random_id}


if __name__ == "__main__":
    url = "https://www.linkedin.com/jobs/view/4308118213/?eBP=NON_CHARGEABLE_CHANNEL&refId=G%2BFao3NOlOSc8QUHNxJkvg%3D%3D&trackingId=aHsjCzSeKWJeghtTUZFbcQ%3D%3D&trk=flagship3_search_srp_jobs&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_jobs%3BN3X1wCt5SMqWow9PKjAieA%3D%3D&lici=aHsjCzSeKWJeghtTUZFbcQ%3D%3D"
    text = get_text_from_url(url)
    # step0 = currenttimemillis()
    # all_skills = read_skills_from_kaggle_file(False)
    # step1 = currenttimemillis()
    # print(f"Step 1 - fetch skills - completed in : {step1 - step0}ms")

    # raw_text = extract_text_from_pdf(
    #     Path("/mnt/g/My Drive/Personal/Resume/Srpincipal/NarayanNatarajan Resume.pdf")
    # )
    # step2 = currenttimemillis()
    # print(f"Step 2 - extract text - completed in : {step2 - step1}ms")
    # resume_skills = extract_skills_from_text_fast(raw_text, all_skills)
    # step3 = currenttimemillis()
    # print(f"Step 3 - extract skills - completed in : {step3 - step2}ms")
    # print(resume_skills)
    # step7= currenttimemillis()
    # print(f"Step 6 - extract skills - completed in : {step7-step6}ms")

    # sys.exit()

    # # Example Usage
    # resume_path = Path("/mnt/c/Users/venkman/Downloads/NarayanNatarajan Resume.pdf")
    # raw_text = extract_text_from_pdf(resume_path)
    # resume_text_path = Path(str(resume_path).replace("/", "_") + ".txt")
    # with open(Path("resumes") / resume_text_path, "w") as f:
    #     f.write(raw_text)
    # # print(raw_text[:200]) # Print first 200 chars
    # # name = extract_name(raw_text)
    # # print(f"Name: {name}")
    # name_email_phone_data = parse_resume_get_name_email_phone(raw_text)
    # print(json.dumps(name_email_phone_data, indent=2))
    # skills_data = parse_resume_get_skills(raw_text)
    # print(json.dumps(skills_data, indent=2))

    # # Example of reading job_skills.csv from the zip file
    # # print("\nJob Skills from Kaggle file:")
    # # print(read_skills_from_kaggle_file())
