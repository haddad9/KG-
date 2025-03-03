# -*- coding: utf-8 -*- #
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command for sending a diagnostic interrupt to an instance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags

DETAILED_HELP = {
    'brief':
        'Send a diagnostic interrupt to a virtual machine instance.',
    'DESCRIPTION':
        """\
          *{command}* is used to send a diagnostic interrupt to a running
          instance, which triggers special interrupt handling logic inside VM.

        For instances with Intel or AMD processors, the guest OS on the instance
        will receive a non-maskable interrupt (NMI).
        """,
    'EXAMPLES':
        """\
        To send a diagnostic interrupt to an instance named ``test-instance'', run:

          $ {command} test-instance
        """
}


@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA,
                    base.ReleaseTrack.GA)
class SendDiagnosticInterrupt(base.SilentCommand):
  """Send a diagnostic interrupt to a virtual machine instance."""

  @staticmethod
  def Args(parser):
    flags.INSTANCE_ARG.AddArgument(parser)

  def Run(self, args):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    client = holder.client

    instance_ref = flags.INSTANCE_ARG.ResolveAsResource(
        args,
        holder.resources,
        scope_lister=flags.GetInstanceZoneScopeLister(client))

    request = client.messages.ComputeInstancesSendDiagnosticInterruptRequest(
        **instance_ref.AsDict())

    return client.MakeRequests([(client.apitools_client.instances,
                                 'SendDiagnosticInterrupt', request)])


SendDiagnosticInterrupt.detailed_help = DETAILED_HELP
