import {Injectable} from '@angular/core';
import {BaseService} from "../base.service";
import {BaseResult, DagStopResult, ProjectAddData} from "../models";
import {AppSettings} from "../app-settings";
import {catchError} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export class ProjectService extends BaseService {
    protected collection_part: string = 'projects';
    protected single_part: string = 'project';


    remove(id: number) {
        let message = `${this.constructor.name}.remove`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/remove';
        return this.http.post<DagStopResult>(url, {'id': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_imgs(id: number) {
        let message = `${this.constructor.name}.remove_imgs`;
        let url = AppSettings.API_ENDPOINT + 'remove_imgs';
        return this.http.post<DagStopResult>(url, {'project': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_files(id: number) {
        let message = `${this.constructor.name}.remove_files`;
        let url = AppSettings.API_ENDPOINT + 'remove_files';
        return this.http.post<DagStopResult>(url, {'project': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    add(data: ProjectAddData) {
        let message = `${this.constructor.name}.add`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/add';
        return this.http.post<BaseResult>(url, data).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    edit(data: ProjectAddData) {
        let message = `${this.constructor.name}.edit`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/edit';
        return this.http.post<BaseResult>(url, data).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }
}