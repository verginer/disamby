# from click.testing import CliRunner
# from disamby import cli
#
#
# def test_command_line_interface():
#     """Test the CLI."""
#     # TODO: use temporary directory to do actual testing
#     runner = CliRunner()
#     result = runner.invoke(cli.main, [])
#     assert result.exit_code == 2  # missing parameters
#     help_result = runner.invoke(cli.main, ['--help'])
#     assert help_result.exit_code == 0
#     assert 'Show this message and exit.' in help_result.output
