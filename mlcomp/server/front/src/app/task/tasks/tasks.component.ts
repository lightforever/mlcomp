import {
    AfterContentChecked,
    Component,
    Input
} from '@angular/core';
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {Paginator} from "../../paginator";
import {TaskFilter} from "../../models";
import {Helpers} from "../../helpers";
import {TaskService} from "../task.service";

@Component({
    selector: 'app-tasks',
    templateUrl: './tasks.component.html',
    styleUrls: ['./tasks.component.css']
})
export class TasksComponent extends Paginator<TasksComponent>
    implements AfterContentChecked {

    ngAfterContentChecked(): void {
        this.route.queryParams.subscribe(params => {
            if (params['dag']) this.dag = parseInt(params['dag']);
            if (params['status']) {
                this.not_ran = false;
                this.queued = false;
                this.in_progress = false;
                this.failed = false;
                this.stopped = false;
                this.skipped = false;
                this.success = false;

                this[params['status']] = true;
            }
        });

    }

    displayed_columns: string[];
    @Input() dag: number;
    name: string;
    id_min: number;
    id_max: number;
    @Input() report: number;

    project: number;

    filter_hidden: boolean = true;
    filter_applied_text: string;

    created_min: string;
    created_max: string;

    not_ran: boolean = false;
    queued: boolean = false;
    in_progress: boolean = false;
    failed: boolean = false;
    stopped: boolean = false;
    skipped: boolean = false;
    success: boolean = false;

    dags: any[];
    projects: any[];

    last_activity_min: string;
    last_activity_max: string;

    dags_model: any[];

    constructor(protected service: TaskService,
                protected location: Location,
                protected router: Router,
                protected  route: ActivatedRoute
    ) {
        super(service,
            location,
            null,
            null,
            true,
            false);
    }

    protected _ngOnInit() {
        let self = this;
        this.data_updated.subscribe(res => {
            if (!res || !res.projects) {
                return;
            }
            self.projects = res.projects;
            self.projects.splice(0, 0,
                {'id': -1, 'name': 'None'});

            self.dags = res.dags;
            self.dags.splice(0, 0,
                {'id': -1, 'name': 'None'});

            self.dags_model = res.dags_model;
        });

    }

    get_filter() {
        let res = new TaskFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        res.report = this.report;

        if (this.project != -1) {
            res.project = this.project;
        }
        if (this.dag != -1) {
            res.dag = this.dag;
        }

        res.status = {
            'not_ran': this.not_ran,
            'queued': this.queued,
            'in_progress': this.in_progress,
            'failed': this.failed,
            'stopped': this.stopped,
            'skipped': this.skipped,
            'success': this.success
        };
        res.created_min = Helpers.parse_time(this.created_min);
        res.created_max = Helpers.parse_time(this.created_max);
        res.last_activity_min = Helpers.parse_time(this.last_activity_min);
        res.last_activity_max = Helpers.parse_time(this.last_activity_max);
        res.id_min = this.id_min;
        res.id_max = this.id_max;

        return res;
    }


    onchange() {
        this.change.emit();

        let filter = this.get_filter();
        let count = 0;
        if (this.dag && this.dag != -1) count += 1;
        if (this.name) count += 1;
        if (this.project && this.project != -1) count += 1;
        if (this.created_min) count += 1;
        if (this.created_max) count += 1;
        if (this.last_activity_min) count += 1;
        if (this.last_activity_max) count += 1;
        for (let k of Object.getOwnPropertyNames(filter.status)) {
            count += filter.status[k] == true ? 1 : 0;
        }
        this.filter_applied_text = count > 0 ? `(${count} applied)` : '';
    }

}