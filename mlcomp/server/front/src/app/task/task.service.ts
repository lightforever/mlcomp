import {Injectable} from '@angular/core';
import {BaseService} from "../base.service";
import {AppSettings} from "../app-settings";
import {catchError} from "rxjs/operators";
import {
    Status,
    StepNodeResult, TaskInfo,
    ToogleReportResult
} from "../models";

@Injectable({
    providedIn: 'root'
})
export class TaskService extends BaseService {
    protected collection_part: string = 'tasks';
    protected single_part: string = 'task';


    stop(id: number) {
        let message = `${this.constructor.name}.stop`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/stop';
        return this.http.post<Status>(url, {'id': id}).pipe(
            catchError(this.handleError<Status>(message, new Status()))
        );
    }

    toogle_report(id: any, report: number, report_full: boolean) {
        let message = `${this.constructor.name}.toogle_report`;
        let url = AppSettings.API_ENDPOINT +
            this.single_part +
            '/toogle_report';

        return this.http.post<ToogleReportResult>(url, {
            'id': id,
            'report': report,
            'remove': report_full
        }).pipe(
            catchError(this.handleError<ToogleReportResult>(message,
                new ToogleReportResult()))
        );
    }

    steps(id: number) {
        let message = `${this.constructor.name}.steps`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/steps';
        return this.http.post<StepNodeResult>(url, id).pipe(
            catchError(this.handleError<StepNodeResult>(message,
                new StepNodeResult()))
        );
    }

    info(id: number) {
        let message = `${this.constructor.name}.info`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/info';
        return this.http.post<TaskInfo>(url, {'id': id}).pipe(
            catchError(this.handleError<TaskInfo>(message, new TaskInfo()))
        );
    }
}
