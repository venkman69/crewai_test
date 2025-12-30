from job_scorev2.crew_analyzer import hr_analyzer_crew
from crew_analyzer import *
from lib import utils
from dotenv import load_dotenv

load_dotenv()


def new_main():
    resume_path = "/home/venkman/git/openai_test/work/resumes/resume_NarayanNatarajan_Resume.pdfn2lukpdk_2025-12-22_09-23-00.pdf.md"
    job_path = "/home/venkman/git/openai_test/work/jobs/_Freddie_Mac_fhJfvz0Br5.txt"
    resume_str = utils.extract_text_from_various_sources(resume_path)
    job_str = utils.extract_text_from_various_sources(job_path)
    result = hr_analyzer_crew(job_str, resume_str, get_from_cache=True)
    print(result)


if __name__ == "__main__":
    new_main()
