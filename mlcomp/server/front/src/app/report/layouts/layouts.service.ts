import {Injectable} from '@angular/core';
import {BaseService} from "../../base.service";
import {AppSettings} from "../../app-settings";
import {BaseResult, LayoutAddData} from "../../models";
import {catchError} from "rxjs/operators";

@Injectable({
  providedIn: 'root'
})
export class LayoutsService extends BaseService{
    protected collection_part: string = 'layouts';
    protected single_part: string = 'layout';

    add(data: LayoutAddData){
        let message = `${this.constructor.name}.add`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/add';
        return this.http.post<BaseResult>(url, data).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    edit(name: string, content: string=null, new_name: string=null){
        let message = `${this.constructor.name}.edit`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/edit';
        let params = {'name': name, 'content': content, 'new_name': new_name};
        return this.http.post<BaseResult>(url, params).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove(name: string) {
        let message = `${this.constructor.name}.remove`;
        let url = AppSettings.API_ENDPOINT + this.single_part + '/remove';
        let params = {'name': name};
        return this.http.post<BaseResult>(url, params).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }
}
