

class AnalysisOutOfOrderError(Exception):

    def __init__(self, required_step: str):
        self.required_step = required_step
        self.message = f"The attempted analysis requires step {required_step} to be executed first"

        super().__init__(self.message)


class MissingAnalysisError(Exception):
    def __init__(self, message: str):
        self.message = message

        super().__init__(self.message)


class ParameterTooStringentError(Exception):

    def __init__(self, parameter_val, parameter_name: str):
        self.relevant_parameter = parameter_name
        self.message = f"Insufficient data after processing; try changing {parameter_name} (current value={parameter_val})"

        super().__init__(self.message)


class OverwritePreviousAnalysisError(Exception):

    def __init__(self, fieldname: str):
        self.fieldname = fieldname
        self.message = f"Should not overwrite field {fieldname}; if this was intended, then set this field to 'None'"

        super().__init__(self.message)


class DataSynchronizationError(Exception):

    def __init__(self, field1: str, field2: str, suggested_method=""):
        self.field1 = field1
        self.field2 = field2
        self.suggested_method = suggested_method
        self.message = f"Fields {field1} and {field2} should be synchronized"
        if len(suggested_method) > 0:
            self.message += f"; try {suggested_method}"

        super().__init__(self.message)


class DeprecationError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)


class NoMatchesError(Exception):
    def __init__(self, message=""):
        self.message = "No matches found; " + message
        super().__init__(self.message)


class NoNeuronsError(Exception):
    def __init__(self, message=""):
        self.message = "No neurons found; " + message
        super().__init__(self.message)


class ShouldBeUnreachableError(Exception):
    def __init__(self, message=""):
        self.message = "This code should be unreachable! " + message
        super().__init__(self.message)


class UnknownValueError(Exception):
    def __init__(self, value=""):
        self.message = f"Unknown value passed: {value}"
        super().__init__(self.message)


class MustBeFiniteError(Exception):
    def __init__(self, value=""):
        self.message = f"Value should not be nan, but was: {value}"
        super().__init__(self.message)


class NoBehaviorAnnotationsError(Exception):
    def __init__(self, message=""):
        if len(message) > 0:
            self.message = "No behavior annotations found; " + message
        else:
            self.message = "No behavior annotations found"
        super().__init__(self.message)


class NoManualBehaviorAnnotationsError(NoBehaviorAnnotationsError):
    def __init__(self, message=""):
        if message is not None:
            self.message = "No manual behavior annotations found; " + message
        else:
            self.message = "No manual behavior annotations found"
        super().__init__(self.message)


class InvalidBehaviorAnnotationsError(Exception):
    def __init__(self, message=""):
        self.message = "Invalid behavior annotations found; " + message
        super().__init__(self.message)


class NeedsAnnotatedNeuronError(Exception):
    def __init__(self, message=""):
        self.message = "Did not find necessary manually annotated neuron(s): " + message
        super().__init__(self.message)


class NoBehaviorDataError(Exception):
    def __init__(self, message=""):
        if len(message) > 0:
            self.message = "No behavior data found; " + message
        else:
            self.message = "No behavior data found"
        super().__init__(self.message)


class IncompleteConfigFileError(Exception):
    def __init__(self, message=""):
        if len(message) > 0:
            self.message = "Missing config file; " + message
        else:
            self.message = "Missing config file"
        super().__init__(self.message)


class TiffFormatError(Exception):
    def __init__(self, message=""):
        self.message = "Error with format of raw tiff file: " + message
        super().__init__(self.message)


class RawDataFormatError(Exception):
    def __init__(self, message=""):
        self.message = "Incorrectly formatted raw data; " + message
        super().__init__(self.message)


class NoNeuropalError(Exception):
    def __init__(self, message=""):
        self.message = "No neuropal data found in project: " + message
        super().__init__(self.message)

class IncorrectNameFormatError(Exception):
    def __init__(self, message=""):
        self.message = "Incorrectly formatted neuron name: " + message
        super().__init__(self.message)
