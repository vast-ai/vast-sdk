from abc import ABC
from typing import Optional, List, Dict, Any


class VastAIBase(ABC):
    """VastAI SDK base class that defines the methods to be implemented by the VastAI class."""

    def attach_ssh(self, instance_id: int, ssh_key: str) -> None:
        """Attach an SSH key to an instance."""
        pass

    def cancel_copy(self, dst: str) -> None:
        """Cancel a file copy operation."""
        pass

    def cancel_sync(self, dst: str) -> None:
        """Cancel a file sync operation."""
        pass

    def change_bid(self, id: int, price: Optional[float] = None) -> None:
        """Change the bid price for a machine."""
        pass

    def copy(self, src: str, dst: str, identity: Optional[str] = None) -> None:
        """Copy files between instances."""
        pass

    def cloud_copy(
        self,
        src: Optional[str] = None,
        dst: Optional[str] = "/workspace",
        instance: Optional[str] = None,
        connection: Optional[str] = None,
        transfer: str = "Instance to Cloud",
    ) -> None:
        """Copy files between cloud and instance."""
        pass

    def create_api_key(
        self,
        name: Optional[str] = None,
        permission_file: Optional[str] = None,
        key_params: Optional[str] = None,
    ) -> None:
        """Create a new API key."""
        pass

    def create_ssh_key(self, ssh_key: str) -> None:
        """Create a new SSH key."""
        pass

    def create_autogroup(
        self,
        test_workers: int = 3,
        gpu_ram: Optional[float] = None,
        template_hash: Optional[str] = None,
        template_id: Optional[int] = None,
        search_params: Optional[str] = None,
        launch_args: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        endpoint_id: Optional[int] = None,
        min_load: Optional[float] = None,
        target_util: Optional[float] = None,
        cold_mult: Optional[float] = None,
    ) -> None:
        """Create a new autoscaler."""
        pass

    def create_endpoint(
        self,
        min_load: float = 0.0,
        target_util: float = 0.9,
        cold_mult: float = 2.5,
        cold_workers: int = 5,
        max_workers: int = 20,
        endpoint_name: Optional[str] = None,
    ) -> None:
        pass

    def create_instance(
        self,
        id: int,
        price: Optional[float] = None,
        disk: Optional[float] = 10,
        image: Optional[str] = None,
        login: Optional[str] = None,
        label: Optional[str] = None,
        onstart: Optional[str] = None,
        onstart_cmd: Optional[str] = None,
        entrypoint: Optional[str] = None,
        ssh: bool = False,
        jupyter: bool = False,
        direct: bool = False,
        jupyter_dir: Optional[str] = None,
        jupyter_lab: bool = False,
        lang_utf8: bool = False,
        python_utf8: bool = False,
        extra: Optional[str] = None,
        env: Optional[str] = None,
        args: Optional[List[str]] = None,
        force: bool = False,
        cancel_unavail: bool = False,
        template_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new instance from a contract offer ID."""
        pass

    def create_instances(
        self,
        ids: List[int],
        template_hash: Optional[str] = None,
        user: Optional[str] = None,
        disk: float = 10,
        image: Optional[str] = None,
        login: Optional[str] = None,
        label: Optional[str] = None,
        onstart: Optional[str] = None,
        onstart_cmd: Optional[str] = None,
        entrypoint: Optional[str] = None,
        ssh: bool = False,
        jupyter: bool = False,
        direct: bool = False,
        jupyter_dir: Optional[str] = None,
        jupyter_lab: bool = False,
        lang_utf8: bool = False,
        python_utf8: bool = False,
        extra: Optional[str] = None,
        env: Optional[str] = None,
        args: Optional[List[str]] = None,
        force: bool = False,
        cancel_unavail: bool = False,
        bid_price: Optional[float] = None,
        create_volume: Optional[int] = None,
        link_volume: Optional[int] = None,
        volume_size: Optional[int] = None,
        mount_path: Optional[str] = None,
        volume_label: Optional[str] = None,
    ) -> None:
        """Create multiple instances from a list of offer IDs."""
        pass

    def create_subaccount(
        self,
        email: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        type: Optional[str] = None,
    ) -> None:
        pass

    def create_team(self, team_name: Optional[str] = None) -> None:
        """Create a new team."""
        pass

    def create_team_role(
        self, name: Optional[str] = None, permissions: Optional[str] = None
    ) -> None:
        """Create a new team role."""
        pass

    def create_template(
        self,
        name: Optional[str] = None,
        image: Optional[str] = None,
        image_tag: Optional[str] = None,
        login: Optional[str] = None,
        env: Optional[str] = None,
        ssh: bool = False,
        jupyter: bool = False,
        direct: bool = False,
        jupyter_dir: Optional[str] = None,
        jupyter_lab: bool = False,
        onstart_cmd: Optional[str] = None,
        search_params: Optional[str] = None,
        disk_space: Optional[str] = None,
    ) -> None:
        """Create a new template."""
        pass

    def delete_api_key(self, id: int) -> None:
        pass

    def delete_ssh_key(self, id: int) -> None:
        pass

    def delete_autoscaler(self, id: int) -> None:
        pass

    def delete_endpoint(self, id: int) -> None:
        pass

    def destroy_instance(self, id: int) -> Dict[str, Any]:
        pass

    def destroy_instances(self, ids: List[int]) -> None:
        pass

    def destroy_team(self) -> None:
        pass

    def detach_ssh(self, instance_id: int, ssh_key_id: str) -> None:
        pass

    def execute(self, id: int, COMMAND: str) -> None:
        """Execute a command on an instance."""
        pass

    def invite_team_member(
        self, email: Optional[str] = None, role: Optional[str] = None
    ) -> None:
        """Invite a new member to the team."""
        pass

    def label_instance(self, id: int, label: str) -> None:
        """Label an instance."""
        pass

    def launch_instance(
        self,
        gpu_name: str,
        num_gpus: str,
        image: str,
        region: str = None,
        disk: float = 16.0,
        limit: int = 3,
        order: str = "score-",
        login: str = None,
        label: str = None,
        onstart: str = None,
        onstart_cmd: str = None,
        entrypoint: str = None,
        ssh: bool = False,
        jupyter: bool = False,
        direct: bool = False,
        jupyter_dir: str = None,
        jupyter_lab: bool = False,
        lang_utf8: bool = False,
        python_utf8: bool = False,
        extra: str = None,
        env: str = None,
        args: list = None,
        force: bool = False,
        cancel_unavail: bool = False,
        template_hash: str = None,
        explain: bool = False,
        raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Launches the top instance from the search offers based on the given parameters.

        Returns:
            Dict[str, Any]: JSON response from the create instance API call.
        """
        pass

    def logs(self, INSTANCE_ID: int, tail: Optional[str] = None) -> None:
        """Retrieve logs for an instance."""
        pass

    def prepay_instance(self, id: int, amount: float) -> None:
        """Prepay for an instance."""
        pass

    def reboot_instance(self, id: int) -> None:
        """Reboot an instance."""
        pass

    def recycle_instance(self, id: int) -> None:
        """Recycle an instance."""
        pass

    def remove_team_member(self, id: int) -> None:
        """Remove a member from the team."""
        pass

    def remove_team_role(self, NAME: str) -> None:
        """Remove a role from the team."""
        pass

    def reports(self, id: int) -> None:
        """Generate reports for a machine."""
        pass

    def reset_api_key(self) -> None:
        """Reset the API key."""
        pass

    def start_instance(self, id: int) -> bool:
        """Start an instance."""
        pass

    def start_instances(self, ids: List[int]) -> None:
        """Start multiple instances."""
        pass

    def stop_instance(self, id: int) -> bool:
        """Stop an instance."""
        pass

    def stop_instances(self, ids: List[int]) -> None:
        """Stop multiple instances."""
        pass

    def search_benchmarks(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for benchmarks based on a query."""
        pass

    def search_invoices(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for invoices based on a query."""
        pass

    def search_offers(
        self,
        type: Optional[str] = None,
        no_default: bool = False,
        new: bool = False,
        limit: Optional[int] = None,
        disable_bundling: bool = False,
        storage: Optional[float] = None,
        order: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for offers based on various criteria."""
        pass

    def search_templates(self, query: Optional[str] = None) -> None:
        """Search for templates based on a query."""
        pass

    def set_api_key(self, new_api_key: str) -> None:
        """Set a new API key."""
        pass

    def set_user(self, file: Optional[str] = None) -> None:
        """Set user parameters from a file."""
        pass

    def ssh_url(self, id: int) -> None:
        """Print the SSH URL for an instance."""
        pass

    def scp_url(self, id: int) -> None:
        """Print the SCP URL for transferring files to/from an instance."""
        pass

    def show_api_key(self, id: int) -> None:
        """Print details of an API key."""
        pass

    def show_api_keys(self) -> Dict[str, Any]:
        """Show all API keys."""
        pass

    def show_ssh_keys(self) -> Dict[str, Any]:
        """Show all SSH keys."""
        pass

    def show_autoscalers(self) -> Dict[str, Any]:
        """Show all autoscalers."""
        pass

    def show_endpoints(self) -> List[Dict[str, Any]]:
        """Show all endpoints."""
        pass

    def show_connections(self) -> List[Dict[str, Any]]:
        """Show all connections."""
        pass

    def show_deposit(self, Id: int) -> None:
        """Print deposit details for an instance."""
        pass

    def show_earnings(
        self,
        quiet: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        machine_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Show earnings information."""
        pass

    def show_invoices(
        self,
        quiet: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        only_charges: bool = False,
        only_credits: bool = False,
        instance_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Show invoice details."""
        pass

    def show_instance(self, id: int) -> Dict[str, Any]:
        """Show details of an instance."""
        pass

    def show_instances(self, quiet: bool = False) -> List[Dict[str, Any]]:
        """Show all instances."""
        pass

    def show_ipaddrs(self) -> List[Dict[str, Any]]:
        """Show IP addresses."""
        pass

    def show_user(self, quiet: bool = False) -> Dict[str, Any]:
        """Show user details."""
        pass

    def show_subaccounts(self, quiet: bool = False) -> List[Dict[str, Any]]:
        """Show all subaccounts of the current user."""
        pass

    def show_team_members(self) -> Dict[str, Any]:
        """Show all team members."""
        pass

    def show_team_role(self, NAME: str) -> None:
        """Print details of a specific team role."""
        pass

    def show_team_roles(self) -> Dict[str, Any]:
        """Show all team roles."""
        pass

    def transfer_credit(self, recipient: str, amount: float) -> None:
        """Transfer credit to another account."""
        pass

    def update_autoscaler(
        self,
        id: int,
        min_load: Optional[float] = None,
        target_util: Optional[float] = None,
        cold_mult: Optional[float] = None,
        test_workers: Optional[int] = None,
        gpu_ram: Optional[float] = None,
        template_hash: Optional[str] = None,
        template_id: Optional[int] = None,
        search_params: Optional[str] = None,
        launch_args: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        endpoint_id: Optional[int] = None,
    ) -> None:
        pass

    def update_endpoint(
        self,
        id: int,
        min_load: Optional[float] = None,
        target_util: Optional[float] = None,
        cold_mult: Optional[float] = None,
        cold_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        endpoint_name: Optional[str] = None,
    ) -> None:
        pass

    def update_team_role(
        self, id: int, name: Optional[str] = None, permissions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update details of a team role."""
        pass

    def update_ssh_key(self, id: int, ssh_key: str) -> None:
        """Update an SSH key."""
        pass

    def generate_pdf_invoices(
        self,
        quiet: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        only_charges: bool = False,
        only_credits: bool = False,
    ) -> None:
        """Generate PDF invoices based on filters."""
        pass

    def cleanup_machine(self, id: int) -> Dict[str, Any]:
        """Clean up a machine's configuration and resources."""
        pass

    def list_machine(
        self,
        id: int,
        price_gpu: Optional[float] = None,
        price_disk: Optional[float] = None,
        price_inetu: Optional[float] = None,
        price_inetd: Optional[float] = None,
        discount_rate: Optional[float] = None,
        min_chunk: Optional[int] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List details of a single machine with optional pricing and configuration parameters."""
        pass

    def list_machines(
        self,
        ids: List[int],
        price_gpu: Optional[float] = None,
        price_disk: Optional[float] = None,
        price_inetu: Optional[float] = None,
        price_inetd: Optional[float] = None,
        discount_rate: Optional[float] = None,
        min_chunk: Optional[int] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List details of multiple machines with optional pricing and configuration parameters."""
        pass

    def remove_defjob(self, id: int) -> None:
        """Remove the default job from a machine."""
        pass

    def set_defjob(
        self,
        id: int,
        price_gpu: Optional[float] = None,
        price_inetu: Optional[float] = None,
        price_inetd: Optional[float] = None,
        image: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> None:
        """Set a default job on a machine with specified parameters."""
        pass

    def set_min_bid(self, id: int, price: Optional[float] = None) -> None:
        """Set the minimum bid price for a machine."""
        pass

    def schedule_maint(
        self, id: int, sdate: Optional[float] = None, duration: Optional[float] = None
    ) -> None:
        """Schedule maintenance for a machine."""
        pass

    def cancel_maint(self, id: int) -> None:
        """Cancel scheduled maintenance for a machine."""
        pass

    def unlist_machine(self, id: int) -> None:
        """Unlist a machine from being available for new jobs."""
        pass

    def show_machines(self, quiet: bool = False, filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve and display a list of machines based on specified criteria.

        Parameters:
        - quiet (bool): If True, limit the output to minimal details such as IDs.
        - filter (str, optional): A string used to filter the machines based on specific criteria.
        """
        pass

    def add_network_disk(self, machines: List[int], mount_point: str, disk_id: Optional[int] = None) -> Dict[str, Any]:
        """Add network disk to physical cluster."""
        pass

    def create_cluster(self, subnet: str, manager_id: int) -> Dict[str, Any]:
        """Create a Vast cluster."""
        pass

    def create_network_volume(self, id: int, size: float = 15, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new network volume."""
        pass

    def create_overlay(self, cluster_id: int, name: str) -> Dict[str, Any]:
        """Create an overlay network on top of a physical cluster."""
        pass

    def create_workergroup(
        self,
        template_hash: Optional[str] = None,
        template_id: Optional[int] = None,
        no_default: bool = False,
        launch_args: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        endpoint_id: Optional[int] = None,
        test_workers: int = 3,
        gpu_ram: Optional[float] = None,
        search_params: Optional[str] = None,
        min_load: Optional[float] = None,
        target_util: Optional[float] = None,
        cold_mult: Optional[float] = None,
        cold_workers: Optional[int] = None,
    ) -> None:
        """Create a new autoscale worker group."""
        pass

    def defrag_machines(self, IDs: List[int]) -> None:
        """Defragment machines."""
        pass

    def delete_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Delete a cluster."""
        pass

    def delete_overlay(self, overlay_identifier: Optional[str] = None) -> Dict[str, Any]:
        """Delete an overlay and remove all associated instances."""
        pass

    def delete_scheduled_job(self, id: int) -> None:
        """Delete a scheduled job."""
        pass

    def delete_workergroup(self, id: int) -> None:
        """Delete a worker group."""
        pass

    def get_wrkgrp_logs(self, id: int, level: int = 1, tail: Optional[int] = None) -> Dict[str, Any]:
        """Fetch logs for a specific serverless worker group."""
        pass

    def join_cluster(self, cluster_id: int, machine_ids: List[int]) -> Dict[str, Any]:
        """Join machines to a cluster."""
        pass

    def join_overlay(self, name: str, instance_id: int) -> Dict[str, Any]:
        """Add an instance to an overlay network."""
        pass

    def list_network_volume(
        self,
        disk_id: int,
        price_disk: float = 0.15,
        end_date: Optional[str] = None,
        size: int = 15,
    ) -> Dict[str, Any]:
        """List disk space for rent as a network volume."""
        pass

    def remove_machine_from_cluster(self, cluster_id: int, machine_id: int, new_manager_id: Optional[int] = None) -> Dict[str, Any]:
        """Remove a machine from a cluster."""
        pass

    def search_network_volumes(
        self,
        no_default: bool = False,
        limit: Optional[int] = None,
        storage: float = 1.0,
        order: str = "score-",
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for network volume offers using custom query."""
        pass

    def show_clusters(self) -> Dict[str, Any]:
        """Show clusters associated with your account."""
        pass

    def show_invoices_v1(
        self,
        invoices: bool = False,
        invoice_type: Optional[List[str]] = None,
        charges: bool = False,
        charge_type: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
        next_token: Optional[str] = None,
        format: str = "table",
        verbose: bool = False,
        latest_first: bool = False,
    ) -> None:
        """Get billing history reports with advanced filtering and pagination."""
        pass

    def show_network_disks(self) -> Dict[str, Any]:
        """Show network disks associated with your account."""
        pass

    def show_overlays(self) -> List[Dict[str, Any]]:
        """Show overlays associated with your account."""
        pass

    def show_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """Show the list of scheduled jobs for the account."""
        pass

    def show_workergroups(self) -> List[Dict[str, Any]]:
        """Display current worker groups."""
        pass

    def take_snapshot(
        self,
        instance_id: str,
        container_registry: str = "docker.io",
        repo: Optional[str] = None,
        docker_login_user: Optional[str] = None,
        docker_login_pass: Optional[str] = None,
        pause: str = "true",
    ) -> None:
        """Take a container snapshot and push to a registry."""
        pass

    def tfa_activate(
        self,
        code: str,
        sms: bool = False,
        secret: Optional[str] = None,
        phone_number: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        """Activate a new 2FA method by verifying the code."""
        pass

    def tfa_delete(
        self,
        id_to_delete: Optional[int] = None,
        code: Optional[str] = None,
        sms: bool = False,
        secret: Optional[str] = None,
        backup_code: Optional[str] = None,
        method_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Remove a 2FA method from your account."""
        pass

    def tfa_login(
        self,
        code: Optional[str] = None,
        sms: bool = False,
        secret: Optional[str] = None,
        backup_code: Optional[str] = None,
        method_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete 2FA login by verifying code."""
        pass

    def tfa_regen_codes(
        self,
        code: Optional[str] = None,
        sms: bool = False,
        secret: Optional[str] = None,
        backup_code: Optional[str] = None,
        method_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Regenerate backup codes for 2FA."""
        pass

    def tfa_resend_sms(self, phone_number: Optional[str] = None, secret: Optional[str] = None) -> None:
        """Resend SMS 2FA code."""
        pass

    def tfa_send_sms(self, phone_number: Optional[str] = None) -> None:
        """Request a 2FA SMS verification code."""
        pass

    def tfa_status(self) -> None:
        """Print the current 2FA status and configured methods."""
        pass

    def tfa_totp_setup(self) -> None:
        """Generate TOTP secret and QR code for Authenticator app setup."""
        pass

    def tfa_update(self, method_id: int, label: Optional[str] = None, set_primary: Optional[str] = None) -> Dict[str, Any]:
        """Update a 2FA method's settings."""
        pass

    def unlist_network_volume(self, id: int) -> Dict[str, Any]:
        """Unlist a network volume offer."""
        pass

    def unlist_volume(self, id: int) -> Dict[str, Any]:
        """Unlist a volume offer."""
        pass

    def update_workergroup(
        self,
        id: int,
        min_load: Optional[float] = None,
        target_util: Optional[float] = None,
        cold_mult: Optional[float] = None,
        cold_workers: Optional[int] = None,
        test_workers: Optional[int] = None,
        gpu_ram: Optional[float] = None,
        template_hash: Optional[str] = None,
        template_id: Optional[int] = None,
        search_params: Optional[str] = None,
        no_default: bool = False,
        launch_args: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        endpoint_id: Optional[int] = None,
    ) -> None:
        """Update an existing autoscale worker group."""
        pass

    def clone_volume(self, source: int, dest: int, size: Optional[float] = None, disable_compression: bool = False) -> Dict[str, Any]:
        """Clone an existing volume to create a new volume with optional size increase."""
        pass

    def create_env_var(self, name: str, value: str) -> None:
        """Create a new user environment variable."""
        pass

    def create_volume(self, id: int, size: float = 15, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new volume from an offer ID."""
        pass

    def delete_env_var(self, name: str) -> None:
        """Delete a user environment variable."""
        pass

    def delete_machine(self, id: int) -> None:
        """Delete a machine if not being used by clients."""
        pass

    def delete_template(self, template_id: Optional[int] = None, hash_id: Optional[str] = None) -> None:
        """Delete a template by template ID or hash ID."""
        pass

    def delete_volume(self, id: int) -> Dict[str, Any]:
        """Delete a volume by ID."""
        pass

    def get_endpt_logs(self, id: int, level: int = 1, tail: Optional[int] = None) -> Dict[str, Any]:
        """Fetch logs for a specific serverless endpoint group."""
        pass

    def invite_member(self, email: str, role: str) -> None:
        """Invite a team member to your current team."""
        pass

    def list_volume(self, id: int, price_disk: float = 0.10, end_date: Optional[str] = None, size: int = 15) -> Dict[str, Any]:
        """List disk space for rent as a volume on a machine."""
        pass

    def list_volumes(self, ids: List[int], price_disk: float = 0.10, end_date: Optional[str] = None, size: int = 15) -> List[Dict[str, Any]]:
        """List disk space for rent as volumes on multiple machines."""
        pass

    def remove_member(self, id: int) -> None:
        """Remove a team member by user ID."""
        pass

    def search_volumes(self, query: Optional[str] = None, storage: float = 1.0, order: str = "score-") -> List[Dict[str, Any]]:
        """Search for volume offers using custom query."""
        pass

    def self_test_machine(
        self,
        machine_id: str,
        debugging: bool = False,
        explain: bool = False,
        raw: bool = False,
        url: str = "https://console.vast.ai",
        retry: int = 3,
        ignore_requirements: bool = False,
    ) -> Dict[str, Any]:
        """Perform a self-test on the specified machine."""
        pass

    def show_audit_logs(self) -> List[Dict[str, Any]]:
        """Display account's history of important actions."""
        pass

    def show_env_vars(self, show_values: bool = False) -> Dict[str, Any]:
        """Show user environment variables."""
        pass

    def show_machine(self, Machine: int, quiet: bool = False) -> Dict[str, Any]:
        """Show details of a hosted machine."""
        pass

    def show_maints(self, ids: str, quiet: bool = False) -> List[Dict[str, Any]]:
        """Show maintenance information for host machines."""
        pass

    def show_members(self) -> Dict[str, Any]:
        """Show your team members."""
        pass

    def show_volumes(self, type: str = "all") -> List[Dict[str, Any]]:
        """Show stats on owned volumes."""
        pass

    def update_env_var(self, name: str, value: str) -> None:
        """Update an existing user environment variable."""
        pass

    def update_instance(
        self,
        id: int,
        template_id: Optional[int] = None,
        template_hash_id: Optional[str] = None,
        image: Optional[str] = None,
        args: Optional[str] = None,
        env: Optional[str] = None,
        onstart: Optional[str] = None,
    ) -> None:
        """Update/recreate an instance from a new/updated template."""
        pass

    def update_template(
        self,
        HASH_ID: str,
        name: Optional[str] = None,
        image: Optional[str] = None,
        env: Optional[str] = None,
        onstart_cmd: Optional[str] = None,
        search_params: Optional[str] = None,
        disk: Optional[float] = None,
        ssh: bool = False,
        direct: bool = False,
        jupyter: bool = False,
    ) -> None:
        """Update an existing template."""
        pass
