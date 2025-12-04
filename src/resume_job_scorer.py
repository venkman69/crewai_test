import os
from crewai import LLM
from lib import utils
from pathlib import Path
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process

# Read your API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Use Gemini 2.5 Pro Experimental model
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.0,  # Lower temperature for more consistent results.
)


class TextExtractor(BaseTool):
    name: str = "text_extractor"
    description: str = "Extracts the text from a any PDF, text file, raw string or url."

    def _run(self, text_source: str) -> str:
        try:
            if Path(text_source).exists():
                if text_source.endswith(".pdf"):
                    print(f"Extracting text from PDF: {text_source}")
                    return utils.extract_text_from_pdf(text_source)
                elif text_source.endswith(".txt"):
                    print(f"Extracting text from TXT: {text_source}")
                    return utils.extract_text_from_txt(text_source)
            elif text_source.startswith("http"):
                print(f"Extracting text from URL: {text_source}")
                return utils.get_text_from_url(text_source)
            else:
                # assume the text_source itself is the text
                raise ValueError(f"Text source is not a PDF or TXT file: {text_source}")
        except Exception as e:
            if len(text_source) > 100 and not text_source.startswith("http"):
                print(
                    f"warning: assuming text source is already text since length is greater than 100 {len(text_source)}: {text_source}"
                )
                return text_source
            else:
                raise ValueError(
                    f"Text source is not a PDF or TXT file and is too short: {text_source}"
                )


class JobSourceIdentifier(BaseTool):
    name: str = "job_source_identifier"
    description: str = "Identifies the source of a URL and identifies a reconstructable job url and ability to create filename"

    def _run(self, job_url: str) -> str:
        return utils.identify_job_source(job_url)


text_extractor = TextExtractor()
job_source_identifier = JobSourceIdentifier()

resume_text_extractor_agent = Agent(
    role="Resume Text extractor",
    goal="""Extract the text from the file or string provided.""",
    backstory="""You are a text extractor and can process raw text, urls, text files and PDFs.""",
    llm=gemini_llm,
    tools=[text_extractor],
    verbose=True,
    allow_delegation=False,
)
resume_text_extraction_task = Task(
    description="""Extract the text from the resume from {resume}.""",
    agent=resume_text_extractor_agent,
    expected_output="""Output should be the raw text from the resume.""",
)
resume_skill_analyzer_agent = Agent(
    role="Resume skill analyzer",
    goal="""Extract the skills from the file or string. """,
    backstory="""You are an expert HR professional specializing in identifying skills within a resume.""",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)
resume_skill_analyzer_task = Task(
    description="""Extract the skills from the resume raw string data.""",
    context=[resume_text_extraction_task],
    agent=resume_skill_analyzer_agent,
    expected_output="""Output should be in JSON format, 
    for example : {"resume_skills":["Python","WAF","Jira"]}""",
)
job_text_extractor_agent = Agent(
    role="Job Text extractor",
    goal="""Extract the text from the file, url or string from provided input. """,
    backstory="""You are a text extractor and can process raw text, urls, text files and PDFs.
    You can also identify the source of a URL and identify a reconstructable job url and ability to create filename.""",
    llm=gemini_llm,
    tools=[text_extractor, job_source_identifier],
    verbose=True,
    allow_delegation=False,
)
job_text_extraction_task = Task(
    description="""Extract the text from the job description from {job_description}.""",
    agent=job_text_extractor_agent,
    expected_output="""Output should be in JSON format, 
    for example : {"job_description":"Job description text","job_posting_details":
    {"job_source": "Indeed", "job_id": job_id, "job_url": "https://www.indeed.com/viewjob?jk=1234567890"}}""",
)

job_skill_analyzer_agent = Agent(
    role="Job skill analyzer",
    goal="""Extract the skills from the job description. """,
    backstory="""You are an expert HR professional specializing in identifying skills 
    within a job description. You can also extract the organization name from the job description.
    In addition you can pass the job posting details.""",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)
job_skill_analyzer_task = Task(
    description="""Extract the organization and skills.
    Skills should be categorized into required and preferred.
    Specifically from the job_description key. 
    Pass on the job_posting_details as is.""",
    context=[job_text_extraction_task],
    agent=job_skill_analyzer_agent,
    expected_output="""Output should be in JSON format, 
    for example : {"organization":"Google","required_skills":["Python","WAF"],"preferred_skills":["Jira","TOGAF"],"job_posting_details":
    {"job_source": "Indeed", "job_id": "1234567890", "job_url": "https://www.indeed.com/viewjob?vjk=1234567890"}}""",
)

job_vs_resume_skill_matching_agent = Agent(
    role="HR Expert who can review the job skills and candidate skills and assign a score",
    goal="""Review the job skills and candidate skills and assign a score.""",
    backstory="""You are an expert HR professional who can determine how a candidate skills match the required and preferred skills in a job description
    """,
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)
job_vs_resume_skill_matching_task = Task(
    description="""Identify a match score for the job and candidate skills.
    Compare the required and preferred skills for the job with the candidate skills and assign a score.
    Use this criteria to calculate the score:
    1. Count the number of required skills in the job that are present in the candidate and divide by the total number of required skills in the job to get the required skill match score.
    2. Count the number of preferred skills in the job that are present in the candidate and divide by the total number of preferred skills in the job to get the preferred skill match score.
    3. Final score = (required skill match score * 0.7 + preferred skill match score * 0.3 ) / 2 
    for example if job has 10 required skills and 5 preferred skills and 
       candidate has 8 required skills and 3 preferred skills then final score = (8/10 * 0.7 + 3/5 * 0.3 ) / 2 = 0.85
    4. Score data should be presented as keys in JSON output as below:
       a. A tag named "score" which is a dictionary with the following keys:
          i. final_score
          ii. required_skill_match_score
          iii. preferred_skill_match_score
          iv. matching_required_skills_count, 
          v. missing_required_skills_count, 
          vi. matching_preferred_skills_count, 
          vii. missing_preferred_skills_count, 
          viii. total_required_skills_count,
          ix.  total_preferred_skills_count, 
    """,
    agent=job_vs_resume_skill_matching_agent,
    expected_output="""Output should be in JSON format, 
    for example : {"matching_required_skills":["Java","Python"], "missing_required_skills":["Ruby", "TOGAF"], 
    "matching_preferred_skills":["CISSP","WAF"],"missing_preferred_skills":["Jira","management"],
    "job_posting_details":
    {"job_source": "Indeed", "job_id": "1234567890", "job_url": "https://www.indeed.com/viewjob?vjk=1234567890"}}""",
    context=[job_skill_analyzer_task, resume_skill_analyzer_task],
)

if __name__ == "__main__":
    do_resume = True
    do_job = True
    do_jobscore = True
    do_caching = False
    resume_analysis_inputs = {
        "resume": "/mnt/g/My Drive/Personal/Resume/Srpincipal/NarayanNatarajan Resume.pdf",
    }
    job_analysis_inputs = {
        "job_description": "https://www.linkedin.com/jobs/view/4295968762/?alternateChannel=search&eBP=CwEAAAGa4PqF9IgNWii4iSjDDPMNXHNpLSA3GECJuI-uGa-Ne-morKHWvEck3u5kVRiyyQBZGFpmOm6eraWrFcR_VVDtinlos0gnLfIbaRsB5GjYm9MCocCb0cprrpNoYsq3-JS1-Pir8UoORkiN0d6xA-PNFSZ-oNQd0IH9DkUPhXhqaugknQW-dcDUD2xp5wLuHCEdePv_eumk1uDxKs0rRaTX-cA31igAAwGnOFDzQLswdxVZ3TiOhKOnprGucJ_KXF3EmPpOYtZDwV-0PCMIpfGh_Q01CPAIuDr8FE1WM8M9_FpQjJecLdwqEuMJsN7bpO-f5p_vjGUn4y_lq6VJMZsGmkD-iryxCAk7VbngmRlh5c_LebO2hdzkmaTTl5DpXHPoi-6E9EjQVB0kZa_ARJKFBG8WNnA4-U-3iv1UM7b3WptxM98BGgmhcikoJA5U3-McWe27BRzX7vY-7g6Pit2gXCIzQxJnnFQQdiFzINXJaClkkIBxeYfZGQiki4-33EE&refId=2zTc0DSmGEO7DB24663Txw%3D%3D&trackingId=6U9WzULLaqRTlHhnCnnS3A%3D%3D",
    }

    if do_resume:
        resume_crew_runner = Crew(
            agents=[resume_text_extractor_agent, resume_skill_analyzer_agent],
            tasks=[resume_text_extraction_task, resume_skill_analyzer_task],
            process=Process.sequential,
            verbose=True,
        )

        if do_caching:
            resume_text_for_caching = text_extractor.run(
                resume_analysis_inputs["resume"]
            )
            if utils.dc.get(resume_text_for_caching):
                print("Resume skills already cached - retrieving from cache")
                resume_result = utils.dc.get(resume_text_for_caching)
            else:
                print(f"Resume skills not cached - running crewai")
                resume_result = resume_crew_runner.kickoff(
                    inputs=resume_analysis_inputs
                )
                utils.dc.set(resume_text_for_caching, resume_result)
        else:
            print(f"Resume skills: do_caching is False ({do_caching}) - running crewai")
            resume_result = resume_crew_runner.kickoff(inputs=resume_analysis_inputs)
            if do_caching:
                utils.dc.set(resume_text_for_caching, resume_result)
        # print("--- Resume Skills Analysis Result ---")
        # print(resume_result)
        with open("resume_skills_analysis.txt", "w") as f:
            f.write(resume_result.raw)
        with open("resume_skills_analysis.json", "w") as f:
            # find location of : ```json
            start = resume_result.raw.find("```json") + len("```json")
            end = resume_result.raw.find("```", start)
            f.write(resume_result.raw[start:end])

    if do_job:
        job_crew_runner = Crew(
            agents=[job_text_extractor_agent, job_skill_analyzer_agent],
            tasks=[job_text_extraction_task, job_skill_analyzer_task],
            process=Process.sequential,
            verbose=True,
        )
        if do_caching:
            job_text_for_caching = text_extractor.run(
                job_analysis_inputs["job_description"]
            )
            if utils.dc.get(job_text_for_caching):
                print("Job skills already cached - retrieving from cache")
                job_result = utils.dc.get(job_text_for_caching)
            else:
                print(f"Job skills not cached - running crewai")
                job_result = job_crew_runner.kickoff(inputs=job_analysis_inputs)

        else:
            print(f"Job skills: do_caching is False ({do_caching}) - running crewai")
            job_result = job_crew_runner.kickoff(inputs=job_analysis_inputs)

        # get the job_id and job_source from job_result and create a filename from it.
        job_result_json = utils.extract_json_from_crew_output(job_result.raw)
        job_id = job_result_json["job_posting_details"]["job_id"]
        job_source = job_result_json["job_posting_details"]["job_source"]
        job_filename = f"job_skills_analysis_{job_source}_{job_id}.txt"
        # print("--- Job Skills Analysis Result ---")
        # print(job_result)
        with open(job_filename, "w") as f:
            f.write(job_result.raw)

    if do_jobscore:
        job_scorer_crew_runner = Crew(
            agents=[job_vs_resume_skill_matching_agent],
            tasks=[job_vs_resume_skill_matching_task],
            process=Process.sequential,
            verbose=True,
        )
        if do_caching:
            if not (job_result and resume_result):
                print(
                    "Cannot Proceed: Missing one or both of job_result and resume_result"
                )
                exit(1)
            job_vs_resume_skill_matching_text_for_caching = (job_result, resume_result)
            if utils.dc.get(job_vs_resume_skill_matching_text_for_caching):
                print(
                    "Job vs Resume skill matching already cached - retrieving from cache"
                )
                job_score_result = utils.dc.get(
                    job_vs_resume_skill_matching_text_for_caching
                )
            else:
                print(f"Job vs Resume skill matching not cached - running crewai")
                job_score_result = job_scorer_crew_runner.kickoff(
                    inputs=job_analysis_inputs
                )
                utils.dc.set(
                    job_vs_resume_skill_matching_text_for_caching, job_score_result
                )
        else:
            print(
                f"Job vs Resume skill matching: do_caching is False ({do_caching}) - running crewai"
            )
            job_score_result = job_scorer_crew_runner.kickoff(
                inputs=job_analysis_inputs
            )
        with open("job_vs_resume_skill_matching.txt", "w") as f:
            f.write(job_score_result.raw)
        with open("job_vs_resume_skill_matching.json", "w") as f:
            # find location of : ```json
            start = job_score_result.raw.find("```json") + len("```json")
            end = job_score_result.raw.find("```", start)
            f.write(job_score_result.raw[start:end])

    print("--- Crew Execution Finished ---")
