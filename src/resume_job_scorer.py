import os
from crewai import LLM
from lib import utils
from pathlib import Path
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process, CrewOutput
import json
import datetime
# from crewai.memory import LongTermMemory

# Read your API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# list of free tier rates:
# gemini/gemini-2.5-flash-lite: RPD=20, RPM=10, TPM=250K
# gemini/gemini-2.5-flash:      RPD=20, RPM=5,  TPM=250K
gemini_flash_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.0,  # Lower temperature for more consistent results.
)
gemini_flash_lite_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=gemini_api_key,
    temperature=0.0,  # Lower temperature for more consistent results.
)
ollama_llm = LLM(
    model="ollama/mistral",
    base_url="http://localhost:11434",
)

openai_llm = LLM(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,  # Lower temperature for more consistent results.
)
llm_dict = {
    "gemini_flash": gemini_flash_llm,
    "gemini_flash_lite": gemini_flash_lite_llm,
    "ollama": ollama_llm,
    "openai": openai_llm,
}
llm_choice_name = "gemini_flash_lite"
print(f"Using LLM: {llm_choice_name}")
llm_choice = llm_dict[llm_choice_name]


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
                    return utils.extract_text_from_file(text_source)
            elif text_source.startswith("http"):
                print(f"Extracting text from URL: {text_source}")
                return utils.get_text_from_url(text_source)
            else:
                return text_source
        except Exception:
            # exception is thrown if it is already text and Path will throw an exception
            print("Assuming the text source is already raw text")
            return text_source


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
    llm=llm_choice,
    tools=[text_extractor],
    verbose=True,
    allow_delegation=False,
    memory=True,
)
resume_skill_analyzer_agent = Agent(
    role="Resume skill analyzer",
    goal="""Extract the skills from the file or string. """,
    backstory="""You are an expert HR professional specializing in identifying skills within a resume.""",
    llm=llm_choice,
    verbose=True,
    allow_delegation=False,
    memory=True,
)
resume_skill_analyzer_task = Task(
    description="""Extract the skills and years of work experience from the resume raw string data within resume text.
    work experience is overall number of years from the start date of the first job to the end date of the last job and round up.
    Resume Text is provided as below: {resume_text}""",
    agent=resume_skill_analyzer_agent,
    expected_output="""Output MUST be just the JSON object, 
    for example : 
        {
            "resume_skills":["Python","WAF","Jira"], 
            "years_of_experience":5,
        }""",
)
job_skill_analyzer_agent = Agent(
    role="Job skill analyzer",
    goal="""Extract the skills from the job description. """,
    backstory="""You are an expert HR professional specializing in identifying skills 
    within a job description. You can also extract the organization name from the job description.
    """,
    llm=llm_choice,
    verbose=True,
    allow_delegation=False,
    memory=True,
)
job_skill_analyzer_task = Task(
    description="""Extract:
    * organization name, 
    * summary of the role, 
    * years of industry experience,
    * and skills categorized into required and preferred.
    from provided inputs in: job_text:{job_text} and job_posting_details:{job_posting_details}.
    Skills should be categorized into required and preferred.
    if a particular skill mentions a minimum years of experience, 
    then represent it within brackets for example "Python (2 years)".
    Add the job_posting_details to the output.
    Add the organization name and years of experience to the job_posting_details.""",
    agent=job_skill_analyzer_agent,
    expected_output="""Output MUST be just the JSON object, 
    for example : 
        {
            "required_skills":["Python (2 years)","WAF"],
            "preferred_skills":["Jira","TOGAF"],
            "job_posting_details":
            {
                "role_summary":"Role summary text",
                "organization":"Google",
                "years_of_experience": "5",
                "job_source": "Indeed", 
                "job_id": "1234567890", 
                "job_url": "https://www.indeed.com/viewjob?vjk=1234567890"
            }
        }""",
)

job_vs_resume_skill_matching_agent = Agent(
    role="HR Expert who can review the job skills and candidate skills and assign a score",
    goal="""Review the job skills and candidate skills and assign a score.""",
    backstory="""You are an expert HR professional who can determine how a candidate skills match the required and preferred skills in a job description
    """,
    llm=llm_choice,
    verbose=True,
    allow_delegation=False,
    memory=True,
)
job_vs_resume_skill_matching_task = Task(
    description="""Identify a match score for the job and candidate skills contained within the JSON formatted strings.
    Compare the required and preferred skills for the job with the candidate skills and assign a score.
    Use this criteria to calculate the score:
    1. Calculate 'required_skill_match_score': Count the number of required skills in the job that are present in the candidate and divide by the total number of required skills in the job to get the required skill match score.
    2. Calculate 'preferred_skill_match_score': Count the number of preferred skills in the job that are present in the candidate and divide by the total number of preferred skills in the job to get the preferred skill match score.
    3. Calculate 'final_score': Final score = required_skill_match_score * 0.7 + preferred_skill_match_score * 0.3  
    for example if job has 10 required skills and 5 preferred skills and 
       candidate has 8 required skills and 3 preferred skills then final score = 8/10 * 0.7 + 3/5 * 0.3 = 0.74
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
    5. Pass the job_posting_details as is
    6. If a skill mentions a minimum years of experience, then compare the years of experience 
        in the resume with the minimum years of experience in the job description. if the years of experience 
        in the resume is less than the minimum years of experience in the job description then consider 
        it as a missing skill.
    Job skils are in: {job_analysis} and candidate skills are in: {resume_analysis}
    
    """,
    agent=job_vs_resume_skill_matching_agent,
    expected_output="""Output should be in JSON format, 
    for example : 
        {
            "matching_required_skills":["Java","Python"], 
            "missing_required_skills":["Ruby", "TOGAF"], 
            "matching_preferred_skills":["CISSP","WAF"],
            "missing_preferred_skills":["Jira","management"],
            "organization":"Google",
            "job_posting_details": {
                "job_source": "Indeed", 
                "job_id": "1234567890", 
                "job_url": "https://www.indeed.com/viewjob?vjk=1234567890"
            },
            "score": {
                "final_score": 0.2139,
                "required_skill_match_score": 0.2667,
                "preferred_skill_match_score": 0.0909,
                "matching_required_skills_count": 8,
                "missing_required_skills_count": 22,
                "matching_preferred_skills_count": 1,
                "missing_preferred_skills_count": 10,
                "total_required_skills_count": 30,
                "total_preferred_skills_count": 11
            }
        }""",
)
job_vs_resume_skill_matching_agent_v2 = Agent(
    role="HR Expert who can review the job description, then extract required and preferred skills and compare with candidate skills and assign a score",
    goal="""Review the job skills and candidate skills and assign a score.""",
    backstory="""You are an expert HR professional who can determine how a candidate skills match the required and preferred skills in a job description
    """,
    llm=llm_choice,
    verbose=True,
    allow_delegation=False,
    memory=True,
)

job_vs_resume_skill_matching_task_v2 = Task(
    description="""# Goal:
    Review the job description, then extract required and preferred skills and compare with candidate skills and assign a score.
    ## Job data extraction:
    You will extract the following information from the job description:
    * organization name,
    * summary of the role,
    * years of industry experience,
    * and skills categorized into required and preferred.

    ## Use this criteria to calculate the score:
    1. Calculate 'required_skill_match_score': Count the number of required skills in the job that are present in the candidate and divide by the total number of required skills in the job to get the required skill match score.
    2. Calculate 'preferred_skill_match_score': Count the number of preferred skills in the job that are present in the candidate and divide by the total number of preferred skills in the job to get the preferred skill match score.
    3. Calculate 'final_score': Final score = required_skill_match_score * 0.7 + preferred_skill_match_score * 0.3  
    for example if job has 10 required skills and 5 preferred skills and 
       candidate has 8 required skills and 3 preferred skills then final score = 8/10 * 0.7 + 3/5 * 0.3 = 0.74
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
    ## Comparison guidance for candidate skills vs job requirements: 
    1. Look at the required missing skills, if any of the following appear then candidate will not pass:
        a. Active Security Clearance.
        b. Specific certifications that are required.
        c. If a skill mentions a minimum years of experience, then compare 
    2. If the candidate does not fail due to any of the above criteria, and the final score is above 70%,
    then the decision should be "Pass" and the reason should be "Score is above 70% and candidate has sufficient skills and years of experience".
    Otherwise the decision should be "Fail" along with the appropriate reason(s).
    3. Provide the final decision and reason in the output.

    # Inputs: 
    * Job description is in: {job_text} 
    * Candidate skills are in JSON format within the resume analysis: {resume_analysis}
    """,
    agent=job_vs_resume_skill_matching_agent,
    expected_output="""Output MUST be just the JSON object, 
    for example : 
        {
            "matching_required_skills":["Java","Python"], 
            "missing_required_skills":["Ruby", "TOGAF"], 
            "matching_preferred_skills":["CISSP","WAF"],
            "missing_preferred_skills":["Jira","management"],
            "score": {
                "final_score": 0.7633,
                "required_skill_match_score": 0.8333,
                "preferred_skill_match_score": 0.60,
                "matching_required_skills_count": 20,
                "missing_required_skills_count": 4,
                "matching_preferred_skills_count": 6,
                "missing_preferred_skills_count": 4,
                "total_required_skills_count": 24,
                "total_preferred_skills_count": 10
            }
            "organization":"Google",
            "years_of_experience":5,
            "job_summary":"Role summary text",
            "decision":"Pass",
            "reason":"Score is above 70% and candidate has sufficient skills and years of experience",
        }""",
)

resume_vs_job_decision_agent = Agent(
    role="""HR Expert who can review the analysis and render a final summary, 
    specifically whether job is a good fit for the candidate""",
    goal="""Check the missing skills to determine if it may disqualify the candidate.""",
    backstory="""You are an expert HR professional who can decide whether a candidate passes
    to the next stage of the hiring process or not.
    """,
    llm=llm_choice,
    verbose=True,
    allow_delegation=False,
    memory=True,
)
resume_vs_job_decision_task = Task(
    description="""
    1. Look at the required missing skills, if any of the following appear then candidate will not pass:
        a. Active Security Clearance.
        b. Specific certifications that are required.
        c. If a skill mentions a minimum years of experience, then compare 
    2. Pass all the data from the job_skill_analyzer_task output.
    3. Add a new key named "resume_vs_job_decision" to the output.
    4. The "resume_vs_job_decision" key should contain a dictionary with the following keys:
        a. decision
        b. reason
    5. If the candidate does not fail due to any of the above criteria, and the final score is above 70%,
    then the decision should be "Pass" and the reason should be "Score is above 70% and candidate has sufficient skills and years of experience".
    Otherwise the decision should be "Fail" along with the appropriate reason(s).
    """,
    agent=resume_vs_job_decision_agent,
    expected_output="""Output should be in JSON format, adding to the output from the resume_vs_job_decision_task. 
    for example this should be added to the output from the resume_vs_job_decision_task within the key "resume_vs_job_decision" :
         {
         "decision":"Pass", 
         "reason":"Score is above 70% and candidate has sufficient skills and years of experience"}
         or
         {
         "decision":"Fail", 
         "reason":"Candidate is missing required skills or Active Security Clearance"}
    """,
)


def resume_skill_analyser_crew(
    resume_url_or_text: str, get_from_cache: bool = True
) -> tuple[CrewOutput, str]:
    # get the resume text
    resume_text = text_extractor.run(resume_url_or_text)
    # append that I am a citizen of the United States
    resume_text += "\nI am a citizen of the United States"
    # extract name, email, phone number from resume text
    candidate_info = utils.nlp_parse_resume_get_name_email_phone(resume_text)
    print(candidate_info)

    if get_from_cache and resume_text in utils.dc:
        print(f"Returning cached result for resume {resume_url_or_text[:100]}...")
        return utils.dc[resume_text]
    resume_crew = Crew(
        agents=[
            resume_skill_analyzer_agent,
        ],
        tasks=[
            resume_skill_analyzer_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
    start = utils.currenttimemillis()
    result = resume_crew.kickoff(inputs={"resume_text": resume_text})
    end = utils.currenttimemillis()
    print(f"Resume skill analysis took {end - start} ms")
    print("Caching result for resume")
    save_path = save_resume_skill_analysis(resume_crew, result, candidate_info)
    utils.dc[resume_text] = result, save_path
    return result, save_path


def job_skill_analyzer_crew(job_url: str, cache: bool = True) -> tuple[CrewOutput, str]:
    # get the job text
    job_text = text_extractor.run(job_url)
    job_details = job_source_identifier.run(job_url)
    if cache and job_text in utils.dc:
        print("Returning cached result for job")
        return utils.dc[job_text]
    job_crew = Crew(
        agents=[
            job_skill_analyzer_agent,
        ],
        tasks=[
            job_skill_analyzer_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
    start = utils.currenttimemillis()
    result = job_crew.kickoff(
        inputs={"job_text": job_text, "job_posting_details": json.dumps(job_details)}
    )
    end = utils.currenttimemillis()
    print(f"Job skill analysis took {end - start} ms")
    print("Caching result for job")
    save_path = save_job_skill_analysis(job_crew, result, "job_skill_analysis")
    utils.dc[job_text] = (result, save_path)
    return result, save_path


def compare_and_decide_crew(
    resume_analysis: str, job_analysis: str, cache: bool = True
) -> tuple[CrewOutput, str]:
    if cache and (resume_analysis, job_analysis) in utils.dc:
        print("Returning cached result for compare and decide")
        return utils.dc[(resume_analysis, job_analysis)]
    compare_and_decide_crew = Crew(
        agents=[
            job_vs_resume_skill_matching_agent,
            resume_vs_job_decision_agent,
        ],
        tasks=[
            job_vs_resume_skill_matching_task,
            resume_vs_job_decision_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
    start = utils.currenttimemillis()
    result = compare_and_decide_crew.kickoff(
        inputs={"resume_analysis": resume_analysis, "job_analysis": job_analysis}
    )
    end = utils.currenttimemillis()
    print(f"Compare and decide took {end - start} ms")
    print("Caching result for compare and decide")
    save_path = save_job_skill_analysis(
        compare_and_decide_crew, result, "final_decision"
    )
    utils.dc[(resume_analysis, job_analysis)] = (result, save_path)
    return result, save_path


def job_analysis_and_decision_crew(
    job_url: str, resume_output: CrewOutput, get_from_cache: bool = True
) -> tuple[CrewOutput, str]:
    # get the job text
    job_text = text_extractor.run(job_url)
    job_details = job_source_identifier.run(job_url)
    resume_analysis = resume_output.raw
    if get_from_cache and (resume_analysis, job_text) in utils.dc:
        print("Returning cached result for compare and decide")
        return utils.dc[(resume_analysis, job_text)]
    job_crew = Crew(
        agents=[
            job_vs_resume_skill_matching_agent_v2,
        ],
        tasks=[
            job_vs_resume_skill_matching_task_v2,
        ],
        process=Process.sequential,
        verbose=True,
    )
    start = utils.currenttimemillis()
    result = job_crew.kickoff(
        inputs={"job_text": job_text, "resume_analysis": resume_analysis}
    )
    end = utils.currenttimemillis()
    print(f"Job vs Resume skill matching took {end - start} ms")
    print("Caching result for job")
    save_path = save_job_skill_analysis(job_crew, result, job_details, "job_vs_resume")
    utils.dc[(resume_analysis, job_text)] = (result, save_path)
    return result, save_path
    pass


def save_job_skill_analysis(
    crew_object: Crew, crew_result: CrewOutput, job_details: dict, prefix: str
):
    print(f"Writing {prefix} task output")
    os.makedirs("logs", exist_ok=True)
    job_skill_analysis_result_json = utils.extract_json_from_crew_output(
        crew_result.raw
    )
    job_org = job_skill_analysis_result_json.get("organization", "Unknown")
    # job_source = {"job_source": "", "job_id": random_id, "job_url": ""}
    job_id = job_details.get("job_id", "Unknown")
    job_source = job_details.get("job_source", "Unknown")
    job_org = job_org.replace(" ", "_")
    job_id = job_id.replace(" ", "_")
    job_source = job_source.replace(" ", "_")
    job_filename = f"{prefix}_{job_source}_{job_org}_{job_id}.txt"
    save_result(crew_object, crew_result, job_filename, job_details)
    return job_filename


def save_resume_skill_analysis(
    crew_object: Crew, crew_result: CrewOutput, candidate_info: dict
):
    print("Writing Resume task outputs to files")
    os.makedirs("logs", exist_ok=True)
    resume_skill_analysis_result_json = utils.extract_json_from_crew_output(
        crew_result.raw
    )
    candidate_name = candidate_info.get("name", "Unknown").replace(" ", "_")
    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resume_analysis_filename = f"resume_skills_analysis_{candidate_name}_{datestr}.txt"
    save_result(crew_object, crew_result, resume_analysis_filename)
    return resume_analysis_filename


def save_result(
    crew_object: Crew, crew_result: CrewOutput, file_name: str, job_details: dict = None
):
    with open(f"logs/{file_name}", "w") as f:
        f.write("Crew Statistics\n")
        f.write(json.dumps(crew_object.usage_metrics.__dict__, indent=4))
        f.write("\n")
        f.write("---------\n")
        for i, task_output in enumerate(crew_result.tasks_output):
            f.write(f"-------------- {task_output.agent} --------------\n")
            f.write(f"Description:\n{task_output.description}\n")
            f.write(f"Summary:\n{task_output.summary}\n")
            for mesg in task_output.messages:
                f.write(f"{mesg['role']}: {mesg['content']}\n")
            f.write(f"Raw:\n{task_output.raw}\n")
            f.write("\n")
        if job_details:
            f.write("Job Details\n")
            f.write(json.dumps(job_details, indent=4))
            f.write("\n")
    return file_name


def scan_parsed_files_dir():
    resume_files = os.listdir("./parsed_files")
    resume_files.sort(
        key=lambda x: os.path.getmtime(os.path.join("./parsed_files", x)), reverse=True
    )
    return resume_files


if __name__ == "__main__":
    # complete_flow_result = run_flow(
    #     "/mnt/g/My Drive/Personal/Resume/Srpincipal/NarayanNatarajan Resume.pdf",
    #     sys.argv[1],
    # )

    # print("--- Crew Execution Finished ---")
    # save_result(complete_flow_result)

    # pick the newest file from resume folder
    # pick google drive file first
    if Path(
        "/mnt/g/My Drive/Personal/Resume/Srpincipal/NarayanNatarajan Resume.pdf"
    ).exists():
        resume = (
            "/mnt/g/My Drive/Personal/Resume/Srpincipal/NarayanNatarajan Resume.pdf"
        )
    else:
        print(
            "** Cannot access google drive file **, checking the parsed_files directory instead"
        )
        resumes = scan_parsed_files_dir()
        if len(resumes) == 0:
            print("No resumes found in parsed_files directory")
            exit(1)
        resume = "./parsed_files/" + resumes[0]
    print(f"Using resume: {resume}")

    # read from command line prompt:
    # job_description = input("Enter job URL: ")
    job_url = "https://www.linkedin.com/jobs/view/4313406950"
    job_url = "https://www.linkedin.com/jobs/view/4247854309"
    job_url = "https://www.linkedin.com/jobs/view/4305691847"
    job_url = "https://www.linkedin.com/jobs/view/4287186320"
    job_url = "https://jobs.baesystems.com/global/en/job/BAE1US115615BREXTERNAL/Chief-Engineer-Cyber-IT-Program?utm_source=linkedin&utm_medium=phenom-feeds"
    job_url = "https://www.linkedin.com/jobs/view/4323936609"
    resume_analysis, resume_analysis_save_path = resume_skill_analyser_crew(
        resume,
        get_from_cache=True,
    )
    # job_analysis, job_analsys_save_path = job_skill_analyzer_crew(
    #     job_url, get_from_cache=True
    # )

    # final_decision, final_decision_save_path = compare_and_decide_crew(
    #     resume_analysis.raw, job_analysis.raw, get_from_cache=True
    # )

    # print(final_decision.raw)
    # print(f"Resume analysis saved to {resume_analysis_save_path}")
    # print(f"Job analysis saved to {job_analsys_save_path}")
    # print(f"Final decision saved to {final_decision_save_path}")

    job_analysis, job_analysis_save_path = job_analysis_and_decision_crew(
        job_url, resume_analysis, get_from_cache=True
    )
    print(job_analysis.raw)
    print(f"Job analysis saved to {job_analysis_save_path}")
