# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

import logging
import traceback
from dataclasses import dataclass
import json

from pydantic import BaseModel, Field

from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT

log = logging.getLogger(__name__)

# these tokens are used in the prompt
INPUT_TEXT_KEY = "input_text"
MAX_LENGTH_KEY = "max_report_length"


class FindingModel(BaseModel):
    """A model for the expected LLM response shape."""

    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    """A model for the expected LLM response shape."""

    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(
        description="A list of findings in the report."
    )
    rating: float = Field(description="The rating of the report.")
    rating_explanation: str = Field(description="An explanation of the rating.")
    community: int | float = Field(description="The community ID this report belongs to.")
    
    def model_post_init(self, __context):
        # Convert float community IDs to integers if they are whole numbers
        if isinstance(self.community, float) and self.community.is_integer():
            self.community = int(self.community)


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: CommunityReportResponse | None
    community: int | None = None


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _model: ChatModel
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int
    _community_id: int | None

    def __init__(
        self,
        model_invoker: ChatModel,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
        community_id: int | None = None,
    ):
        """Init method definition."""
        self._model = model_invoker
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500
        self._community_id = community_id

    async def __call__(self, input_text: str):
        """Call method definition.
        
        Args:
            input_text: The input text to generate a community report from
            
        Returns:
            CommunityReportsResult containing the generated report
        """
        # Convert community_id to int if it's a string or None
        community_id = int(self._community_id) if self._community_id is not None else 0
        log.debug(f"Using community_id: {community_id} (type: {type(community_id).__name__})")
        
        try:
            prompt = self._extraction_prompt.format(**{
                INPUT_TEXT_KEY: input_text,
                MAX_LENGTH_KEY: str(self._max_report_length),
            })
            
            # Log the prompt being sent to the LLM
            log.debug(f"Sending prompt to LLM (first 500 chars):\n{prompt[:500]}...")
            
            try:
                log.info(f"Sending prompt to LLM (first 500 chars):\n{prompt[:500]}...")
                
                # First, get the raw response without JSON parsing
                try:
                    raw_response = await self._model.achat(
                        prompt,
                        json=False,  # Get raw response first
                        name="create_community_report",
                    )
                    
                    # Log the raw response
                    log.info(f"Raw response from LLM (first 2000 chars):\n{str(raw_response)[:2000]}...")
                    if hasattr(raw_response, 'text'):
                        log.info(f"Response text (first 2000 chars):\n{raw_response.text[:2000]}...")
                    
                    # Now try to parse as JSON
                    try:
                        # If the response has a text attribute, use that, otherwise use the string representation
                        response_text = raw_response.text if hasattr(raw_response, 'text') else str(raw_response)
                        
                        # Extract JSON from markdown code blocks if present
                        if '```' in response_text:
                            log.info("Found markdown code blocks in response, extracting JSON content...")
                            # Extract content between first ```json and next ```
                            if '```json' in response_text:
                                json_content = response_text.split('```json')[1].split('```')[0].strip()
                            else:
                                # If no ```json marker, just get content between first set of ```
                                json_content = response_text.split('```')[1].strip()
                            log.info(f"Extracted JSON content: {json_content[:500]}...")
                        else:
                            json_content = response_text
                        
                        # Try to parse the response as JSON
                        import json
                        json_response = json.loads(json_content)
                        log.info("Successfully parsed response as JSON")
                        
                        # Ensure community ID is included in the response before validation
                        if isinstance(json_response, dict):
                            json_response['community'] = community_id
                            log.debug(f"Set community_id in JSON response: {json_response['community']}")
                        
                        # Convert to CommunityReportResponse model
                        response = CommunityReportResponse.model_validate(json_response)
                        
                    except json.JSONDecodeError as je:
                        log.error(f"Failed to parse response as JSON: {str(je)}")
                        log.error(f"Response content that failed to parse: {response_text[:2000]}")
                        raise ValueError(f"LLM response is not valid JSON: {str(je)}") from je
                    except Exception as e:
                        log.error(f"Error validating response against model: {str(e)}")
                        log.error(f"Response content that failed validation: {response_text[:2000]}")
                        raise
                        
                except Exception as e:
                    log.error(f"Error in LLM communication: {str(e)}", exc_info=True)
                    if hasattr(e, 'response'):
                        log.error(f"Error response object: {e.response}")
                        if hasattr(e.response, 'text'):
                            log.error(f"Error response text (first 2000 chars):\n{e.response.text[:2000]}")
                    raise
                
            except Exception as e:
                log.error(f"Error calling LLM: {str(e)}")
                log.error(f"Error type: {type(e).__name__}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    log.error(f"Error response text: {e.response.text}")
                raise
            
            # Initialize default output with community ID
            default_output_data = {
                'title': "Generated Report",
                'summary': "No content available",
                'findings': [],
                'rating': 0.0,
                'rating_explanation': "No rating available",
                'community': self._community_id or 0
            }
            default_output = CommunityReportResponse.model_validate(default_output_data)
            
            # Check if we have a valid response
            if response is None:
                log.warning("Received None response from model")
                return self._create_result(default_output)
                
            # Log the parsed response structure
            log.debug(f"Response type: {type(response)}")
            log.debug(f"Response attributes: {dir(response)}")
            
            # Handle different response formats from AIME API
            if hasattr(response, 'parsed_response') and response.parsed_response is not None:
                # Handle parsed response from JSON mode
                output = response.parsed_response
                log.debug("Successfully parsed response from parsed_response")
            elif hasattr(response, 'text') and response.text:
                response_text = response.text.strip()
                if not response_text:
                    log.warning("Received empty response text")
                    return self._create_result(default_output)
                    
                try:
                    log.debug(f"Attempting to parse response text: {response_text[:200]}...")
                    
                    # Handle markdown code block JSON
                    if '```' in response_text:
                        # Extract content between first ```json and next ```
                        if '```json' in response_text:
                            json_content = response_text.split('```json')[1].split('```')[0].strip()
                        else:
                            # If no ```json marker, just get content between first set of ```
                            json_content = response_text.split('```')[1].strip()
                        log.debug(f"Extracted JSON content: {json_content[:500]}...")
                    else:
                        json_content = response_text
                    
                    # Try to parse the response as JSON
                    parsed = json.loads(json_content)
                    log.debug(f"Successfully parsed JSON: {parsed}")
                    
                    # Handle case where response is wrapped in a 'response' key
                    if isinstance(parsed, dict) and 'response' in parsed:
                        parsed = parsed['response']
                    
                    # Handle both list and dict response formats
                    if isinstance(parsed, list):
                        if len(parsed) > 0:
                            parsed = parsed[0]
                        else:
                            log.warning("Received empty list in response")
                            return self._create_result(default_output)
                    
                    # Ensure we have a dictionary to work with
                    if not isinstance(parsed, dict):
                        log.warning(f"Unexpected response format: {type(parsed).__name__}")
                        return self._create_result(default_output)
                    
                    # Ensure community ID is included in the response before validation
                    parsed['community'] = community_id
                    log.debug(f"Set community_id in parsed response: {parsed['community']}")
                    
                    # Log the final parsed response before validation
                    log.debug(f"Final parsed response before validation: {parsed}")
                    
                    # Validate the response against the model
                    try:
                        output = CommunityReportResponse.model_validate(parsed)
                        log.debug("Successfully validated response against model")
                    except Exception as e:
                        log.error(f"Error validating response against model: {str(e)}")
                        # Instead of returning default output, try to extract what we can from the parsed response
                        if isinstance(parsed, dict):
                            # Create a new response with available fields
                            partial_response = {
                                'title': parsed.get('title', 'Generated Report'),
                                'summary': parsed.get('summary', ''),
                                'findings': parsed.get('findings', []),
                                'rating': parsed.get('rating', 0.0),
                                'rating_explanation': parsed.get('rating_explanation', 'No rating available'),
                                'community': community_id
                            }
                            try:
                                output = CommunityReportResponse.model_validate(partial_response)
                                log.debug("Successfully created partial response from available fields")
                            except Exception as e2:
                                log.error(f"Error creating partial response: {str(e2)}")
                                return self._create_result(default_output)
                        else:
                            return self._create_result(default_output)
                except json.JSONDecodeError as je:
                    log.error(f"Failed to parse response as JSON: {str(je)}")
                    # Try to extract content from the raw response
                    if hasattr(response, 'text'):
                        response_text = response.text
                        # Try to find JSON-like content
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}')
                        if start_idx >= 0 and end_idx > start_idx:
                            try:
                                json_content = response_text[start_idx:end_idx+1]
                                parsed = json.loads(json_content)
                                if isinstance(parsed, dict):
                                    parsed['community'] = community_id
                                    try:
                                        output = CommunityReportResponse.model_validate(parsed)
                                        log.debug("Successfully created response from extracted JSON")
                                    except Exception as e:
                                        log.error(f"Error validating extracted JSON: {str(e)}")
                                        return self._create_result(default_output)
                                else:
                                    return self._create_result(default_output)
                            except Exception as e:
                                log.error(f"Error processing extracted JSON: {str(e)}")
                                return self._create_result(default_output)
                    return self._create_result(default_output)
                except Exception as e:
                    log.error(f"Error processing response: {str(e)}")
                    return self._create_result(default_output)
            else:
                log.warning("No valid response format found (missing both parsed_response and text)")
                return self._create_result(default_output)
            
            # Ensure the output has the correct community ID
            if not hasattr(output, 'community') or output.community is None:
                output.community = community_id
                log.debug(f"Set community_id in output: {output.community}")
            elif output.community != community_id:
                log.warning(f"Output had different community_id: {output.community}, updating to: {community_id}")
                output.community = community_id
                
            return self._create_result(output)
            
        except Exception as e:
            log.exception(f"Error generating community report for community_id={community_id}")
            self._on_error(e, traceback.format_exc(), None)
            # Return a default result with error information and community ID
            error_output_data = {
                'title': "Error Generating Report",
                'summary': f"An error occurred: {str(e)}",
                'findings': [],
                'rating': 0.0,
                'rating_explanation': "Report generation failed",
                'community': community_id  # Use the processed community_id
            }
            error_output = CommunityReportResponse.model_validate(error_output_data)
            return self._create_result(error_output)

    def _create_result(self, output: CommunityReportResponse | None) -> CommunityReportsResult:
        """Helper method to create a consistent CommunityReportsResult.
        
        Args:
            output: The CommunityReportResponse to create a result from, or None
            
        Returns:
            CommunityReportsResult containing the output and metadata
        """
        # Handle None output
        if output is None:
            output = CommunityReportResponse.model_validate({
                'title': "Generated Report",
                'summary': "No content available",
                'findings': [],
                'rating': 0.0,
                'rating_explanation': "No rating available",
                'community': getattr(self, '_community_id', 0) or 0
            })
        
        # Ensure output has community ID set
        if hasattr(output, 'community') and output.community is None:
            output.community = getattr(self, '_community_id', 0) or 0
        
        # Generate text output
        text_output = self._get_text_output(output)
        
        # If text output is empty or error, try to create a basic report from available fields
        if text_output.startswith("Error:") or not text_output.strip():
            findings_text = "\n\n".join(
                f"## {f.summary}\n\n{f.explanation}"
                for f in (output.findings or [])
                if hasattr(f, 'summary') and hasattr(f, 'explanation')
            )
            text_output = f"# {output.title}\n\n{output.summary}\n\n{findings_text}"
        
        # Log the final output for debugging
        log.debug(f"Created result with title: {output.title}")
        log.debug(f"Summary length: {len(output.summary)}")
        log.debug(f"Number of findings: {len(output.findings)}")
        log.debug(f"Text output length: {len(text_output)}")
        
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
            community=output.community
        )

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        """Convert the report to a formatted text output."""
        if not report:
            return "Error: Invalid report format"
            
        try:
            # Format findings
            findings_text = "\n\n".join(
                f"## {f.summary}\n\n{f.explanation}" 
                for f in (report.findings or []) 
                if hasattr(f, 'summary') and hasattr(f, 'explanation')
            )
            
            # Format rating section
            rating_text = f"\n\nImpact Severity Rating: {report.rating}/10\n{report.rating_explanation}"
            
            # Combine all sections
            return f"# {getattr(report, 'title', 'Untitled Report')}\n\n{getattr(report, 'summary', '')}{rating_text}\n\n{findings_text}"
        except Exception as e:
            log.error(f"Error formatting report: {e}")
            # Try to create a basic report from available fields
            try:
                findings_text = "\n\n".join(
                    f"## {f.summary}\n\n{f.explanation}"
                    for f in (report.findings or [])
                    if hasattr(f, 'summary') and hasattr(f, 'explanation')
                )
                return f"# {getattr(report, 'title', 'Untitled Report')}\n\n{getattr(report, 'summary', '')}\n\n{findings_text}"
            except Exception as e2:
                log.error(f"Error creating basic report: {e2}")
                return "Error: Could not format report"
