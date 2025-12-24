from crew_analyzer import *
from lib import utils
from dotenv import load_dotenv
import crew_analyzer
import json
import datetime

load_dotenv()


def new_main():
    resume_path = "/home/venkman/git/openai_test/work/resumes/resume_NarayanNatarajan_Resume.pdfn2lukpdk_2025-12-22_09-23-00.pdf.md"
    job_path = "/home/venkman/git/openai_test/work/jobs/_Freddie_Mac_fhJfvz0Br5.txt"
    resume_result, resume_save_path, resume_crew_usage_metrics = (
        crew_analyzer.resume_skill_analyzer_crew(resume_path, get_from_cache=True)
    )
    job_result, job_save_path, job_details, job_crew_usage_metrics = (
        crew_analyzer.job_requirements_analyzer_crew(job_path, get_from_cache=True)
    )
    hr_result, hr_save_path, hr_job_details, hr_crew_usage_metrics = (
        crew_analyzer.job_vs_resume_analyzer_crew(
            resume_result, job_result, job_details, get_from_cache=True
        )
    )
    print(hr_result.raw)
    print(resume_result)
    print(job_result)


if __name__ == "__main__":
    new_main()
