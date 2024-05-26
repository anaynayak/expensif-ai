""" Use Apple's Vision Framework via PyObjC to perform text detection on images (macOS 10.15+ only) """

import logging
import sys
from typing import List, Optional

import objc
import Quartz
from Cocoa import NSURL
from Foundation import NSDictionary
import Vision

__all__ = ["detect_text", "make_request_handler"]


def detect_text(img_path: str, orientation: Optional[int] = None) -> List:

    with objc.autorelease_pool():
        input_url = NSURL.fileURLWithPath_(img_path)
        input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)

        vision_options = NSDictionary.dictionaryWithDictionary_({})
        if orientation is None:
            vision_handler = (
                Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                    input_image, vision_options
                )
            )
        elif 1 <= orientation <= 8:
            vision_handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_orientation_options_(
                input_image, orientation, vision_options
            )
        else:
            raise ValueError("orientation must be between 1 and 8")
        results = []
        handler = make_request_handler(results)
        vision_request = (
            Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        )
        error = vision_handler.performRequests_error_([vision_request], None)
        vision_request.dealloc()
        vision_handler.dealloc()

        for result in results:
            result[0] = str(result[0])

        return results


def make_request_handler(results):
    """results: list to store results"""
    if not isinstance(results, list):
        raise ValueError("results must be a list")

    def handler(request, error):
        if error:
            print(f"Error! {error}")
        else:
            observations = request.results()
            for text_observation in observations:
                recognized_text = text_observation.topCandidates_(1)[0]
                results.append([recognized_text.string(), recognized_text.confidence()])

    return handler
